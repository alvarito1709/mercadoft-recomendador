#!/usr/bin/env python3
"""
Paso 1 del recomendador de MercadoFT: armado de datasets y EDA.

Lee los CSV exportados de la base (uno por tabla), arma:
  - eventos.csv    un renglón por interacción (usuario pseudonimizado)
  - productos.csv  catálogo con categorías, texto, precio y flag de elegibilidad
  - usuarios.csv   universo de usuarios (solo id pseudonimizado y fecha de registro)
  - eda_resumen.json  métricas del EDA (sin datos personales: se puede compartir)
  - hist_interacciones_por_usuario.png, long_tail_productos.png

Uso:
  pip install pandas numpy matplotlib
  python eda_recomendador.py --data-dir ./csv --out ./out

Los CSV deben llamarse como la tabla (producto.csv, producto_visto.csv, ...). Si el
exportador agrega prefijos o fechas al nombre (mercadoft_producto_visto_2026.csv),
igual los encuentra.
"""
import argparse
import hashlib
import json
import math
import os
import secrets
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLAS_CONOCIDAS = [
    "banner", "busqueda", "calificacion", "calificacion_emprendedor", "categoria",
    "categoria_producto_borrado", "compartido", "consulta", "device_token",
    "emprendedor", "favorito", "producto", "producto_borrado", "producto_categoria",
    "producto_eliminado_imagenes", "producto_imagenes", "producto_tipo_envio",
    "producto_visto", "revision_producto", "revision_producto_motivo", "rol",
    "tipo_envio", "usuario",
]

# Pesos iniciales de cada señal (se ajustan en el paso 3 / evaluación)
PESOS = {"vista": 1.0, "compartido": 2.0, "favorito": 3.0, "calificacion": 3.0, "contacto": 5.0}
RATING_MINIMO_POSITIVO = 4


# ----------------------------------------------------------------------------- utilidades
def find_csv(data_dir: Path, tabla: str):
    archivos = [p for p in data_dir.glob("*.csv")]
    exactos = [p for p in archivos if p.stem.lower() == tabla]
    if exactos:
        return exactos[0]
    cand = [p for p in archivos if tabla in p.stem.lower()]
    # descartar los que en realidad pertenecen a una tabla con nombre más largo
    # que contiene a esta (producto -> producto_visto)
    largas = [t for t in TABLAS_CONOCIDAS if t != tabla and tabla in t]
    cand = [p for p in cand if not any(t in p.stem.lower() for t in largas)]
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        sys.exit(f"Hay más de un CSV que parece ser '{tabla}': {[p.name for p in cand]}. Renombrá el correcto a {tabla}.csv")
    return None


def leer(data_dir: Path, tabla: str, obligatoria: bool, avisos: list):
    p = find_csv(data_dir, tabla)
    if p is None:
        msg = f"No encontré el CSV de la tabla '{tabla}'"
        if obligatoria:
            sys.exit("ERROR: " + msg)
        avisos.append(msg + " (se sigue sin esa fuente)")
        print("AVISO:", msg)
        return None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(p, sep=None, engine="python", encoding=enc,
                             na_values=["NULL", "null", "\\N"])
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  {tabla:<20} {len(df):>7} filas  ({p.name})")
    return df


def to_bool(s: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return False
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, bytes):
            x = x.decode("latin-1")
        t = str(x).strip().lower()
        # los BIT(1) de MySQL a veces salen como el carácter de control \x01 crudo
        return t in {"1", "1.0", "true", "t", "yes", "si", "sí"} or "\x01" in t or "x01" in t
    return s.map(f).astype(bool)


def crudos(s, n=4):
    """Valores crudos más frecuentes de una columna (para diagnosticar formatos)."""
    return {repr(k): int(v) for k, v in s.value_counts(dropna=False).head(n).items()}


def num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def hash_id(x, salt):
    return hashlib.sha256((salt + str(int(x))).encode()).hexdigest()[:12]


def obtener_salt(out: Path) -> str:
    s = os.environ.get("RECO_SALT")
    if s:
        return s
    f = out / ".salt"
    if f.exists():
        return f.read_text().strip()
    s = secrets.token_hex(16)
    f.write_text(s)
    return s


# ----------------------------------------------------------------------------- productos
def armar_productos(d, avisos, resumen):
    prod = leer(d, "producto", True, avisos)
    prod["id"] = num(prod, "id")
    prod = prod.dropna(subset=["id"]).copy()
    prod["id"] = prod["id"].astype(int)

    resumen.setdefault("valores_crudos_bit", {})
    for c in ("activa", "desactivada"):
        if c in prod:
            resumen["valores_crudos_bit"][f"producto.{c}"] = crudos(prod[c])
    activa = to_bool(prod["activa"]) if "activa" in prod else pd.Series(True, index=prod.index)
    desact = to_bool(prod["desactivada"]) if "desactivada" in prod else pd.Series(False, index=prod.index)
    estado = prod["estado_revision"].astype(str).str.upper() if "estado_revision" in prod else pd.Series("APROBADO", index=prod.index)
    # 'activa' puede ser una columna heredada que quedó en 0 para todos: en ese caso se ignora
    usa_activa = bool(activa.any())
    if len(prod) and not usa_activa:
        avisos.append("La columna 'activa' es 0 en todas las filas: se ignora. La elegibilidad usa 'desactivada' y 'estado_revision'")
    base = activa if usa_activa else pd.Series(True, index=prod.index)
    prod["elegible"] = base & ~desact & (estado == "APROBADO")

    resumen["productos_flags"] = {
        "total": int(len(prod)),
        "activa_true": int(activa.sum()),
        "activa_ignorada": not usa_activa,
        "desactivada_true": int(desact.sum()),
        "estado_revision": {k: int(v) for k, v in estado.value_counts().items()},
        "elegibles": int(prod["elegible"].sum()),
    }

    # categorías: categoria_id (principal) + tabla producto_categoria (muchos a muchos)
    cats = {}
    pc = leer(d, "producto_categoria", False, avisos)
    pc_map = {}
    if pc is not None:
        for pid, cid in zip(num(pc, "producto_id"), num(pc, "categoria_id")):
            if pd.notna(pid) and pd.notna(cid):
                pc_map.setdefault(int(pid), set()).add(int(cid))
    inconsistentes = 0
    for pid, cid in zip(prod["id"], num(prod, "categoria_id")):
        s = set(pc_map.get(pid, set()))
        if pd.notna(cid):
            if pc is not None and int(cid) not in s:
                inconsistentes += 1
            s.add(int(cid))
        cats[pid] = s
    resumen["categorias"] = {
        "productos_con_mas_de_una": int(sum(len(v) > 1 for v in cats.values())),
        "productos_sin_categoria": int(sum(len(v) == 0 for v in cats.values())),
        "categoria_id_ausente_en_producto_categoria": int(inconsistentes) if pc is not None else None,
        "cantidad_de_categorias_por_producto": {
            "1": int(sum(len(v) == 1 for v in cats.values())),
            "2": int(sum(len(v) == 2 for v in cats.values())),
            "3": int(sum(len(v) == 3 for v in cats.values())),
            "4_o_mas": int(sum(len(v) >= 4 for v in cats.values())),
        },
    }
    nombres_cat = {}
    cat = leer(d, "categoria", False, avisos)
    if cat is not None:
        nombres_cat = {int(i): str(n) for i, n in zip(num(cat, "id"), cat.get("nombre", "")) if pd.notna(i)}

    emp = leer(d, "emprendedor", False, avisos)
    emp_owner = {}
    if emp is not None and "usuario_id" in emp:
        emp_owner = {int(i): int(u) for i, u in zip(num(emp, "id"), num(emp, "usuario_id")) if pd.notna(i) and pd.notna(u)}
    emp_estado = {}
    if emp is not None and "estado_aprobacion" in emp:
        emp_estado = {int(i): str(e) for i, e in zip(num(emp, "id"), emp["estado_aprobacion"]) if pd.notna(i)}

    productos = pd.DataFrame({
        "producto_id": prod["id"],
        "origen": "producto",
        "nombre": prod.get("nombre", ""),
        "descripcion": prod.get("descripcion", ""),
        "precio": num(prod, "precio"),
        "categorias": prod["id"].map(lambda i: "|".join(map(str, sorted(cats[i])))),
        "categorias_nombres": prod["id"].map(lambda i: "|".join(nombres_cat.get(c, str(c)) for c in sorted(cats[i]))),
        "emprendedor_id": num(prod, "emprendedor_id"),
        "emprendedor_estado": num(prod, "emprendedor_id").map(lambda e: emp_estado.get(int(e), "") if pd.notna(e) else ""),
        "estado_revision": estado,
        "fecha_carga": pd.to_datetime(prod["fecha_carga"], errors="coerce") if "fecha_carga" in prod else pd.NaT,
        "elegible": prod["elegible"],
    })
    if emp_estado:
        resumen["productos_flags"]["elegibles_con_emprendedor_no_aprobado"] = int(
            ((productos["elegible"]) & (productos["emprendedor_estado"] != "APROBADO")).sum())

    # productos borrados: sirven para entender gustos (perfil), nunca se recomiendan
    pb = leer(d, "producto_borrado", False, avisos)
    if pb is not None:
        pb = pb.copy()
        pb["id"] = num(pb, "producto_original_id")
        pb = pb.dropna(subset=["id"])
        pb["id"] = pb["id"].astype(int)
        pb = pb[~pb["id"].isin(productos["producto_id"])]
        borr = pd.DataFrame({
            "producto_id": pb["id"],
            "origen": "borrado",
            "nombre": pb.get("nombre", ""),
            "descripcion": pb.get("descripcion", ""),
            "precio": num(pb, "precio"),
            "categorias": num(pb, "categoria_id").map(lambda c: str(int(c)) if pd.notna(c) else ""),
            "categorias_nombres": num(pb, "categoria_id").map(lambda c: nombres_cat.get(int(c), str(int(c))) if pd.notna(c) else ""),
            "emprendedor_id": num(pb, "emprendedor_id"),
            "emprendedor_estado": "",
            "estado_revision": "BORRADO",
            "fecha_carga": pd.to_datetime(pb["fecha_carga"], errors="coerce") if "fecha_carga" in pb else pd.NaT,
            "elegible": False,
        })
        productos = pd.concat([productos, borr], ignore_index=True)
    resumen["productos_flags"]["borrados_recuperados"] = int((productos["origen"] == "borrado").sum())
    owner = {int(p): emp_owner.get(int(e)) for p, e in zip(productos["producto_id"], productos["emprendedor_id"])
             if pd.notna(e) and int(e) in emp_owner}
    return productos, owner


# ----------------------------------------------------------------------------- eventos
def eventos_desde(df, tipo, col_fecha, resumen_fuentes, filtro=None):
    if df is None:
        return None
    df = df.copy()
    if filtro is not None:
        df = df[filtro(df)]
    pid = num(df, "producto_id")
    orig = num(df, "producto_original_id")
    clave = pid.fillna(orig)
    usuario = num(df, "usuario_id")
    fecha = pd.to_datetime(df[col_fecha], errors="coerce") if col_fecha in df else pd.Series(pd.NaT, index=df.index)
    resumen_fuentes[tipo] = {
        "filas": int(len(df)),
        "sin_usuario": int(usuario.isna().sum()),
        "sin_producto": int(clave.isna().sum()),
        "producto_id_nulo_pero_con_original": int((pid.isna() & orig.notna()).sum()),
        "sin_fecha": int(fecha.isna().sum()),
    }
    return pd.DataFrame({"usuario_id": usuario, "producto_id": clave, "tipo": tipo, "fecha": fecha})


def armar_eventos(d, avisos, resumen):
    fuentes = {}
    partes = []

    partes.append(eventos_desde(leer(d, "producto_visto", True, avisos), "vista", "fecha", fuentes))
    partes.append(eventos_desde(leer(d, "favorito", False, avisos), "favorito", "fecha_creado", fuentes))
    partes.append(eventos_desde(leer(d, "consulta", False, avisos), "contacto", "fecha", fuentes))

    cal = leer(d, "calificacion", False, avisos)
    if cal is not None:
        total = len(cal)
        positivas = cal[num(cal, "calificacion") >= RATING_MINIMO_POSITIVO]
        partes.append(eventos_desde(positivas, "calificacion", "fecha", fuentes))
        fuentes["calificacion"]["filas_totales_antes_de_filtrar_rating"] = int(total)

    comp = leer(d, "compartido", False, avisos)
    if comp is not None:
        mask = (lambda x: x["tipo"].astype(str).str.upper() == "PRODUCTO") if "tipo" in comp else None
        partes.append(eventos_desde(comp, "compartido", "fecha", fuentes, filtro=mask))

    ev = pd.concat([p for p in partes if p is not None], ignore_index=True)
    resumen["fuentes"] = fuentes

    antes = len(ev)
    ev = ev.dropna(subset=["usuario_id", "producto_id"]).copy()
    ev["usuario_id"] = ev["usuario_id"].astype(int)
    ev["producto_id"] = ev["producto_id"].astype(int)
    resumen["eventos_descartados_sin_usuario_o_producto"] = int(antes - len(ev))
    ev["peso"] = ev["tipo"].map(PESOS)
    return ev


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    d = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    salt = obtener_salt(out)
    avisos = []
    resumen = {"avisos": avisos, "pesos_iniciales": PESOS}

    print("Leyendo CSV:")
    productos, owner = armar_productos(d, avisos, resumen)
    ev = armar_eventos(d, avisos, resumen)

    # usuarios
    usr = leer(d, "usuario", False, avisos)
    universo = None
    eliminados = set()
    if usr is not None:
        usr["id"] = num(usr, "id")
        usr = usr.dropna(subset=["id"]).copy()
        usr["id"] = usr["id"].astype(int)
        for c in ("eliminado", "email_verificado"):
            if c in usr:
                resumen.setdefault("valores_crudos_bit", {})[f"usuario.{c}"] = crudos(usr[c])
        elim = to_bool(usr["eliminado"]) if "eliminado" in usr else pd.Series(False, index=usr.index)
        verif = to_bool(usr["email_verificado"]) if "email_verificado" in usr else pd.Series(True, index=usr.index)
        eliminados = set(usr.loc[elim, "id"])
        universo = usr[~elim & verif].copy()
        resumen["usuarios"] = {
            "total": int(len(usr)),
            "eliminados": int(elim.sum()),
            "no_eliminados": int((~elim).sum()),
            "no_eliminados_y_verificados": int(len(universo)),
            "por_rol_id": {str(k): int(v) for k, v in usr.loc[~elim, "rol_id"].value_counts().items()} if "rol_id" in usr else {},
        }
        if len(universo) == 0:
            avisos.append("El universo de usuarios quedó vacío: revisá valores_crudos_bit['usuario.email_verificado']")
    n_antes = len(ev)
    ev = ev[~ev["usuario_id"].isin(eliminados)].copy()
    resumen["eventos_de_usuarios_eliminados_excluidos"] = int(n_antes - len(ev))

    # eventos de un emprendedor sobre sus propios productos (no son señal de gusto)
    dueno = ev["producto_id"].map(owner)
    propios = dueno.notna() & (dueno == ev["usuario_id"])
    resumen["eventos_sobre_producto_propio_excluidos"] = int(propios.sum())
    ev = ev[~propios].copy()

    # concentración: ¿hay cuentas de prueba o usuarios que dominan los datos?
    por_u = ev.groupby("usuario_id").size().sort_values(ascending=False)
    rol_map = {}
    if usr is not None and "rol_id" in usr:
        rol_map = {int(i): (int(r) if pd.notna(r) else None) for i, r in zip(usr["id"], num(usr, "rol_id"))}
    resumen["concentracion_usuarios"] = {
        "top5_pct_de_eventos": round(100 * float(por_u.head(5).sum() / por_u.sum()), 1),
        "top5": [{"eventos": int(n), "rol_id": rol_map.get(int(u))} for u, n in por_u.head(5).items()],
        "usuarios_con_mas_de_100_eventos": int((por_u > 100).sum()),
        "mediana_eventos_por_usuario": float(por_u.median()),
    }

    # --- pares usuario-producto (para el EDA) ---
    pares = ev.groupby(["usuario_id", "producto_id"]).agg(
        peso=("peso", "max"), n_eventos=("tipo", "size")).reset_index()
    ids_elegibles = set(productos.loc[productos["elegible"], "producto_id"])
    ids_catalogo = set(productos["producto_id"])
    pares["producto_elegible"] = pares["producto_id"].isin(ids_elegibles)
    pares["producto_en_catalogo"] = pares["producto_id"].isin(ids_catalogo)

    resumen["eventos"] = {
        "total": int(len(ev)),
        "por_tipo": {k: int(v) for k, v in ev["tipo"].value_counts().items()},
        "usuarios_con_algun_evento": int(ev["usuario_id"].nunique()),
        "pares_usuario_producto": int(len(pares)),
        "pares_sobre_producto_no_elegible": int((~pares["producto_elegible"]).sum()),
        "pares_sobre_producto_fuera_de_catalogo": int((~pares["producto_en_catalogo"]).sum()),
        "fecha_min": str(ev["fecha"].min()),
        "fecha_max": str(ev["fecha"].max()),
        "eventos_por_mes": {str(k): int(v) for k, v in ev["fecha"].dropna().dt.to_period("M").value_counts().sort_index().items()},
        "eventos_por_semana": {str(k): int(v) for k, v in ev["fecha"].dropna().dt.to_period("W").value_counts().sort_index().items()},
    }
    v = ev[ev["tipo"] == "vista"].sort_values(["usuario_id", "producto_id", "fecha"])
    delta = v.groupby(["usuario_id", "producto_id"])["fecha"].diff().dt.total_seconds()
    resumen["eventos"]["vistas_repetidas_en_menos_de_10s"] = int((delta < 10).sum())
    resumen["eventos"]["vistas_repetidas_en_menos_de_60s"] = int((delta < 60).sum())

    # --- cold start / distribución por usuario ---
    prods_por_usuario = pares.groupby("usuario_id")["producto_id"].nunique()
    if universo is not None:
        ids_universo = set(universo["id"]) | set(prods_por_usuario.index)
    else:
        ids_universo = set(prods_por_usuario.index)
        avisos.append("Sin tabla usuario: el % de cold start no incluye usuarios sin ninguna interacción")
    conteo = prods_por_usuario.reindex(sorted(ids_universo)).fillna(0).astype(int)
    labels = ["0", "1", "2", "3-5", "6-10", "11+"]
    dist = pd.cut(conteo, bins=[-1, 0, 1, 2, 5, 10, 10**9], labels=labels).value_counts().reindex(labels)
    resumen["usuarios_por_cantidad_de_productos_distintos"] = {
        "universo": int(len(conteo)),
        "distribucion": {k: int(v) for k, v in dist.items()},
        "mediana": float(conteo.median()),
        "media": round(float(conteo.mean()), 2),
        "cold_start_pct_segun_umbral": {
            f"menos_de_{u}_productos": round(100 * float((conteo < u).mean()), 1) for u in (2, 3, 4, 5)},
        "usuarios_evaluables_leave_one_out_2plus": int((conteo >= 2).sum()),
    }

    # --- productos ---
    cnt_prod = pares[pares["producto_elegible"]].groupby("producto_id")["usuario_id"].nunique()
    cnt_prod = cnt_prod.reindex(sorted(ids_elegibles)).fillna(0).astype(int)
    if len(cnt_prod):
        orden = cnt_prod.sort_values(ascending=False).values
        k = max(1, math.ceil(0.10 * len(orden)))
        total_i = orden.sum()
        usuarios_activos = pares["usuario_id"].nunique()
        resumen["productos_elegibles"] = {
            "cantidad": int(len(cnt_prod)),
            "sin_ninguna_interaccion_pct": round(100 * float((cnt_prod == 0).mean()), 1),
            "con_1_o_2_usuarios_pct": round(100 * float(cnt_prod.between(1, 2).mean()), 1),
            "mediana_usuarios_por_producto": float(cnt_prod.median()),
            "media_usuarios_por_producto": round(float(cnt_prod.mean()), 2),
            "top10pct_productos_concentran_pct_de_pares": round(100 * float(orden[:k].sum() / total_i), 1) if total_i else None,
        }
        pares_el = int(pares["producto_elegible"].sum())
        resumen["densidad_matriz"] = {
            "sobre_universo_completo_pct": round(100 * pares_el / (len(conteo) * len(cnt_prod)), 3),
            "sobre_usuarios_con_interaccion_pct": round(100 * pares_el / (usuarios_activos * len(cnt_prod)), 3),
        }

    # --- contacto como señal objetivo ---
    contactos = ev[ev["tipo"] == "contacto"]
    set_c = contactos.groupby("usuario_id")["producto_id"].agg(set)
    set_todos = pares.groupby("usuario_id")["producto_id"].agg(set)
    con_otro = sum(1 for u, sc in set_c.items() if len(set_todos[u] - sc) >= 1)
    resumen["contacto"] = {
        "eventos": int(len(contactos)),
        "usuarios_que_contactaron": int(len(set_c)),
        "productos_distintos_contactados": int(contactos["producto_id"].nunique()),
        "usuarios_que_contactaron_2_o_mas_productos_distintos": int(sum(len(x) >= 2 for x in set_c)),
        "usuarios_con_contacto_y_algun_producto_no_contactado": int(con_otro),
        "contactos_sobre_producto_visto_antes_por_el_mismo_usuario": int(
            contactos.merge(ev[ev["tipo"] == "vista"][["usuario_id", "producto_id"]].drop_duplicates(),
                            on=["usuario_id", "producto_id"]).shape[0]),
    }

    # --- gráficos ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, dist.values, color="#2563eb")
    for i, v in enumerate(dist.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Productos distintos con los que interactuó el usuario")
    ax.set_ylabel("Cantidad de usuarios")
    ax.set_title("Distribución de interacciones por usuario")
    fig.tight_layout()
    fig.savefig(out / "hist_interacciones_por_usuario.png", dpi=150)
    plt.close(fig)

    if len(cnt_prod):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(orden) + 1), orden, color="#7c3aed")
        ax.set_xlabel("Productos elegibles (ordenados por popularidad)")
        ax.set_ylabel("Usuarios distintos que interactuaron")
        ax.set_title("Cola larga del catálogo")
        fig.tight_layout()
        fig.savefig(out / "long_tail_productos.png", dpi=150)
        plt.close(fig)

    # --- archivos de salida (pseudonimizados) ---
    ev_out = ev.copy()
    ev_out["usuario"] = ev_out["usuario_id"].map(lambda x: hash_id(x, salt))
    ev_out = ev_out[["usuario", "producto_id", "tipo", "fecha", "peso"]].sort_values("fecha")
    ev_out.to_csv(out / "eventos.csv", index=False)
    productos.to_csv(out / "productos.csv", index=False)
    if universo is not None:
        u_out = pd.DataFrame({
            "usuario": universo["id"].map(lambda x: hash_id(x, salt)),
            "fecha_registro": pd.to_datetime(universo["fecha_de_registro"], errors="coerce").dt.date
            if "fecha_de_registro" in universo else None})
        u_out.to_csv(out / "usuarios.csv", index=False)

    with open(out / "eda_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("\n===== RESUMEN (también en eda_resumen.json) =====")
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nListo. Archivos en: {out.resolve()}")


if __name__ == "__main__":
    main()
