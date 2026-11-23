#!/usr/bin/env python3
"""
Pasos 2 y 3 del recomendador de MercadoFT: modelo basado en contenido (con señal de búsqueda opcional),
baselines y evaluación.

Entrada: --in (salida del paso 1: eventos.csv, productos.csv) y --data-dir (busqueda.csv, opcional).
Salida (carpeta --out):
  resultados_evaluacion.json   todo lo medido (sin datos personales)
  resultados_tabla.csv         tabla plana para el escrito
  comparacion_modelos.png      Hit@10 por modelo con intervalos de confianza
  modelo.joblib                modelo final entrenado con todos los datos (lo usa la API del paso 5)

Uso:
  uv run python modelo_eval.py --data-dir data --in out --out out

Protocolos (todos respetan el tiempo: perfil, búsquedas y popularidad se calculan solo con lo ocurrido ANTES
de la primera interacción con el producto objetivo; candidatos = elegibles ya publicados, sin los que el
usuario ya conocía):
  A_val   se deja afuera el penúltimo producto nuevo de cada usuario
  A_test  se deja afuera el último producto nuevo de cada usuario
  B       se deja afuera el último producto contactado (proxy de éxito de la hipótesis)

Selección de configuración (bloques texto/categoría/precio, esquema y peso de búsqueda w_b):
por NDCG@10 sobre A_val + A_test juntos (A_todos). Como esos casos ya se usaron para elegir, A_todos es
optimista para el modelo de contenido; B no participa de ninguna selección. Además se mide la estabilidad
de la selección repitiendo "elegir en la mitad de los usuarios, medir en la otra mitad".
"""
import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import joblib
from mft import busqueda as bq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=UserWarning)

STOP_ES = [
    "de", "la", "el", "en", "y", "a", "los", "las", "del", "al", "con", "para", "por", "un", "una",
    "unos", "unas", "que", "se", "su", "sus", "lo", "es", "o", "como", "mas", "más", "muy", "sin",
    "sobre", "este", "esta", "estos", "estas", "tu", "te", "me", "mi", "nuestro", "nuestra",
]
# (peso texto, peso categorías, peso precio)
GRID_BLOQUES = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (1, 0.5, 0.3), (1, 1, 0.3), (0, 1, 0), (0, 1, 0.3)]
ESQUEMAS = ["pesos", "binario"]
UMBRALES = [1, 2, 3, 4, 5, 6, 8]
SEGMENTOS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 10**9)]
PESOS_BUSQUEDA = [0, 0.5, 1, 2, 3]
# Configuración final (decisión documentada en el notebook): las configuraciones con y sin categoría/precio no se
# distinguen estadísticamente (ver 'seleccion_configuracion' en el JSON); se fija la que usa los tres bloques por
# estabilidad entre muestras y por ser explicable. Con --auto se usa en cambio el máximo de NDCG@10 en A_todos.
CONFIG_FINAL = {"pesos_bloques": (1, 0.5, 0.3), "esquema": "pesos", "peso_busqueda": 2, "umbral": 1}
SEMILLA = 42


# ----------------------------------------------------------------------------- datos
def cargar(d: Path):
    ev = pd.read_csv(d / "eventos.csv", parse_dates=["fecha"]).dropna(subset=["fecha"])
    prod = pd.read_csv(d / "productos.csv", parse_dates=["fecha_carga"],
                       dtype={"categorias": str, "categorias_nombres": str})
    for c in ("nombre", "descripcion", "categorias", "categorias_nombres"):
        prod[c] = prod[c].fillna("").astype(str)
    prod["elegible"] = prod["elegible"].astype(str).str.lower().isin(["true", "1"])
    prod = prod.drop_duplicates("producto_id").reset_index(drop=True)
    return ev, prod


def armar_pares(ev):
    pares = ev.groupby(["usuario", "producto_id"]).agg(
        peso=("peso", "max"), n=("tipo", "size"), t0=("fecha", "min")).reset_index()
    c = (ev[ev["tipo"] == "contacto"].groupby(["usuario", "producto_id"])["fecha"].max()
         .rename("t_contacto").reset_index())
    return pares.merge(c, how="left", on=["usuario", "producto_id"])


# ----------------------------------------------------------------------------- features
def bloques_features(prod):
    texto = (prod["nombre"] + " ") * 2 + prod["descripcion"] + " " + prod["categorias_nombres"].str.replace("|", " ", regex=False)
    tfidf = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=STOP_ES,
                            ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
    T = tfidf.fit_transform(texto)

    cats = sorted({c for s in prod["categorias"] for c in s.split("|") if c})
    ci = {c: i for i, c in enumerate(cats)}
    filas, cols = [], []
    for r, s in enumerate(prod["categorias"]):
        for c in s.split("|"):
            if c:
                filas.append(r)
                cols.append(ci[c])
    C = sp.csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(len(prod), max(len(cats), 1)))

    precio = np.log1p(prod["precio"].fillna(prod["precio"].median()).clip(lower=0).to_numpy())
    b = pd.qcut(precio, 5, labels=False, duplicates="drop")
    P = sp.csr_matrix(pd.get_dummies(b).to_numpy(dtype=float))
    return normalize(T), normalize(C), normalize(P), tfidf, cats


def armar_X(T, C, P, w):
    X = sp.hstack([w[0] * T, w[1] * C, w[2] * P]).tocsr()
    return normalize(X)


# ----------------------------------------------------------------------------- casos
def construir_casos(pares, idx, eleg_ids, modo):
    casos = []
    for u, g in pares.groupby("usuario"):
        g = g.sort_values("t0")
        if modo == "B":
            c = g[g["t_contacto"].notna() & g["producto_id"].isin(eleg_ids)].sort_values("t_contacto")
            if c.empty:
                continue
            tgt = c.iloc[-1]
        else:
            ge = g[g["producto_id"].isin(eleg_ids)]
            k = 1 if modo == "A_val" else 0
            if len(ge) < k + 1:
                continue
            tgt = ge.iloc[-(k + 1)]
        prof = g[(g["t0"] < tgt["t0"]) & g["producto_id"].isin(list(idx.keys()))]
        if prof.empty or int(tgt["producto_id"]) not in idx:
            continue
        casos.append({
            "usuario": u,
            "tgt": idx[int(tgt["producto_id"])],
            "t": tgt["t0"],
            "prof_rows": np.array([idx[int(p)] for p in prof["producto_id"]]),
            "prof_w": prof["peso"].to_numpy(dtype=float),
            "n_prof": int(len(prof)),
        })
    return casos


def candidatos(caso, eleg_rows, carga):
    t = np.datetime64(caso["t"])
    ok = np.isnat(carga[eleg_rows]) | (carga[eleg_rows] <= t)
    cand = np.setdiff1d(eleg_rows[ok], caso["prof_rows"])
    forzado = False
    if caso["tgt"] not in cand:
        cand = np.append(cand, caso["tgt"])
        forzado = True
    return cand, forzado


# ----------------------------------------------------------------------------- modelos
class Contexto:
    def __init__(self, pares, idx, n_prod):
        u_idx = {u: i for i, u in enumerate(pares["usuario"].unique())}
        self.n_prod = n_prod
        self.n_users = len(u_idx)
        self.u_row = pares["usuario"].map(u_idx).to_numpy()
        self.pid_row = pares["producto_id"].map(idx).to_numpy()
        self.t0 = pares["t0"].to_numpy()
        self.peso = pares["peso"].to_numpy(dtype=float)


def score_random(caso, cand, ctx, **kw):
    return np.zeros(len(cand))


def score_popularidad(caso, cand, ctx, **kw):
    m = ctx.t0 < np.datetime64(caso["t"])
    pop = np.bincount(ctx.pid_row[m], weights=ctx.peso[m], minlength=ctx.n_prod)
    return pop[cand]


def score_contenido(caso, cand, ctx, X=None, esquema="pesos", **kw):
    w = caso["prof_w"] if esquema == "pesos" else np.ones(len(caso["prof_w"]))
    u = sp.csr_matrix(w.reshape(1, -1)) @ X[caso["prof_rows"]]
    return (X[cand] @ u.T).toarray().ravel()


def score_knn(caso, cand, ctx, esquema="pesos", **kw):
    m = ctx.t0 < np.datetime64(caso["t"])
    M = sp.csr_matrix((np.ones(int(m.sum())), (ctx.u_row[m], ctx.pid_row[m])),
                      shape=(ctx.n_users, ctx.n_prod)).tocsc()
    norms = np.sqrt(np.asarray(M.sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    G = (M[:, caso["prof_rows"]].T @ M[:, cand]).toarray()
    G = G / norms[caso["prof_rows"]][:, None] / norms[cand][None, :]
    w = caso["prof_w"] if esquema == "pesos" else np.ones(len(caso["prof_w"]))
    return (w[:, None] * G).sum(axis=0)


# ----------------------------------------------------------------------------- métricas
def metricas_bloque(g, e):
    """Métricas esperadas cuando el objetivo está en un bloque de e empates precedido por g mejores."""
    pos = np.arange(g + 1, g + e + 1)
    out = {}
    for K in (5, 10):
        out[f"hit{K}"] = float(np.mean(pos <= K))
    out["ndcg10"] = float(np.mean(np.where(pos <= 10, 1.0 / np.log2(pos + 1), 0.0)))
    out["mrr"] = float(np.mean(1.0 / pos))
    return out


def evaluar(casos, scorer, ctx, eleg_rows, carga, rng, **kw):
    filas, tops, n_forz = [], [], 0
    for c in casos:
        cand, forz = candidatos(c, eleg_rows, carga)
        n_forz += int(forz)
        s = scorer(c, cand, ctx, **kw)
        st = s[np.where(cand == c["tgt"])[0][0]]
        g, e = int((s > st).sum()), int((s == st).sum())
        filas.append({**metricas_bloque(g, e), "n_prof": c["n_prof"], "n_cand": len(cand)})
        orden = np.lexsort((rng.random(len(s)), -s))
        tops.append(cand[orden[:10]])
    df = pd.DataFrame(filas)
    df.attrs["top10"] = tops
    df.attrs["forzados"] = n_forz
    return df


def combinar(df_contenido, df_pop, umbral):
    usar = (df_contenido["n_prof"] >= umbral).to_numpy()
    out = df_pop.copy()
    out.loc[usar, :] = df_contenido.loc[usar, :]
    out.attrs["top10"] = [a if u else b for u, a, b in zip(usar, df_contenido.attrs["top10"], df_pop.attrs["top10"])]
    return out


def ic(x, rng, B=2000):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [None, None]
    medias = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return [round(float(np.percentile(medias, 2.5)), 4), round(float(np.percentile(medias, 97.5)), 4)]


def resumir(df, n_eleg, cola_rows, rng, con_ic=True):
    top = np.concatenate(df.attrs["top10"]) if len(df) else np.array([], dtype=int)
    r = {
        "n_casos": int(len(df)),
        "hit5": round(float(df["hit5"].mean()), 4),
        "hit10": round(float(df["hit10"].mean()), 4),
        "ndcg10": round(float(df["ndcg10"].mean()), 4),
        "mrr": round(float(df["mrr"].mean()), 4),
        "cobertura10": round(len(set(top.tolist())) / max(n_eleg, 1), 4),
        "pct_cola_top10": round(float(np.isin(top, cola_rows).mean()), 4) if len(top) else None,
    }
    if con_ic:
        r["hit10_ic95"] = ic(df["hit10"], rng)
        r["ndcg10_ic95"] = ic(df["ndcg10"], rng)
    return r


def dif_pareada(a, b, rng):
    d = (a - b).to_numpy()
    return {"dif": round(float(d.mean()), 4), "ic95": ic(d, rng)}


# ----------------------------------------------------------------------------- modelo final
def guardar_modelo(destino, prod, ctx, X, tfidf, cats, nombre_a_id, n_precio, wb_blk, esq, umbral, wb_b):
    """Modelo que usa la API (src/mft/recomendador.py). Se entrena con todos los datos."""
    pop_total = np.bincount(ctx.pid_row, weights=ctx.peso, minlength=len(prod))
    joblib.dump({
        "version": str(date.today()),
        "producto_ids": prod["producto_id"].astype(int).to_numpy(),
        "elegible": prod["elegible"].to_numpy(),
        "X": X,
        "popularidad": pop_total,
        "config": {"pesos_bloques": wb_blk, "esquema": esq, "umbral": umbral, "peso_busqueda": wb_b},
        "pesos_por_tipo": {"vista": 1.0, "compartido": 2.0, "favorito": 3.0, "calificacion": 3.0, "contacto": 5.0},
        # para vectorizar búsquedas en la API (usar la misma versión de scikit-learn: ver uv.lock)
        "busqueda": {"tfidf": tfidf, "cats": cats, "nombre_cat_a_id": nombre_a_id, "n_precio": int(n_precio)},
    }, destino)


# ----------------------------------------------------------------------------- main
def media(df, col):
    return round(float(df[col].mean()), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="out")
    ap.add_argument("--out", default="out")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--auto", action="store_true",
                    help="usar la configuración de máximo NDCG@10 en A_todos en lugar de CONFIG_FINAL")
    ap.add_argument("--solo-entrenar", action="store_true",
                    help="entrena con CONFIG_FINAL y escribe modelo.joblib sin evaluar (modo del job nocturno)")
    ap.add_argument("--splits", type=int, default=300, help="repeticiones del análisis de estabilidad")
    args = ap.parse_args()
    din, dout = Path(args.inp), Path(args.out)
    dout.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    res = {"avisos": []}

    ev, prod = cargar(din)
    pares = armar_pares(ev)
    idx = {int(p): i for i, p in enumerate(prod["producto_id"])}
    eleg_rows = np.where(prod["elegible"].to_numpy())[0]
    eleg_ids = set(prod.loc[prod["elegible"], "producto_id"].astype(int))
    carga = prod["fecha_carga"].to_numpy(dtype="datetime64[ns]")
    ctx = Contexto(pares, idx, len(prod))

    usuarios_por_prod = pares.groupby("producto_id")["usuario"].nunique()
    cola_ids = set(usuarios_por_prod[usuarios_por_prod <= 2].index) | (eleg_ids - set(usuarios_por_prod.index))
    cola_rows = np.array([idx[p] for p in cola_ids if p in idx and p in eleg_ids])

    T, C, P, tfidf, cats = bloques_features(prod)
    res["features"] = {"productos": int(len(prod)), "elegibles": int(len(eleg_rows)),
                       "vocabulario_tfidf": int(len(tfidf.vocabulary_)), "categorias": int(len(cats))}

    nombre_a_id = bq.mapa_categorias(prod)
    if args.solo_entrenar:
        cfg = CONFIG_FINAL
        X = armar_X(T, C, P, cfg["pesos_bloques"])
        guardar_modelo(dout / "modelo.joblib", prod, ctx, X, tfidf, cats, nombre_a_id, P.shape[1],
                       cfg["pesos_bloques"], cfg["esquema"], cfg["umbral"], cfg["peso_busqueda"])
        print(f"Modelo entrenado sin evaluar: {len(prod)} productos ({len(eleg_rows)} elegibles) -> {dout / 'modelo.joblib'}")
        return

    # --- señal de búsqueda ---
    busq, binfo = bq.cargar_busquedas(Path(args.data_dir), dout, set(ev["usuario"]))
    if busq is None:
        res["avisos"].append(f"Sin señal de búsqueda: {binfo}")
        print("AVISO: sin señal de búsqueda:", binfo)
        terminos, term_idx, Tq, Cq, pesos_b = [], {}, None, None, [0]
    else:
        terminos = sorted(busq["termino"].unique())
        term_idx = {t: i for i, t in enumerate(terminos)}
        Tq, Cq = bq.vectores_busqueda(terminos, nombre_a_id, tfidf, cats)
        sin_match = (np.asarray(Tq.sum(axis=1)).ravel() == 0) & (np.asarray(Cq.sum(axis=1)).ravel() == 0)
        binfo["terminos_sin_vocabulario_ni_categoria"] = int(sin_match.sum())
        pesos_b = PESOS_BUSQUEDA
    res["busquedas"] = binfo

    casos = {m: construir_casos(pares, idx, eleg_ids, m) for m in ("A_val", "A_test", "B")}
    for m in casos:
        bq.adjuntar(casos[m], busq, term_idx)
    res["casos"] = {m: len(v) for m, v in casos.items()}
    res["casos_con_busqueda_previa"] = {m: int(sum(len(c["q_rows"]) > 0 for c in v)) for m, v in casos.items()}
    print("Casos:", res["casos"], "| con búsqueda previa:", res["casos_con_busqueda_previa"])
    casos_all = casos["A_val"] + casos["A_test"]
    n_val = len(casos["A_val"])
    casos_final = {"A_todos": casos_all, "B": casos["B"]}

    # --- 1) elegir configuración (bloques, esquema, w_b) con A_val + A_test ---
    grid, claves, nd_casos = [], [], []
    for w in GRID_BLOQUES:
        X = armar_X(T, C, P, w)
        Q = bq.matriz_Q(Tq, Cq, w, P.shape[1]) if busq is not None else None
        for esq in ESQUEMAS:
            for wb in pesos_b:
                df = evaluar(casos_all, bq.score_contenido_busq, ctx, eleg_rows, carga, rng,
                             X=X, Q=Q, wb=wb, esquema=esq)
                nd = df["ndcg10"].to_numpy()
                claves.append((w, esq, wb))
                nd_casos.append(nd)
                grid.append({"pesos_bloques": {"texto": w[0], "categoria": w[1], "precio": w[2]},
                             "esquema": esq, "peso_busqueda": wb,
                             "ndcg10_val": round(float(nd[:n_val].mean()), 4),
                             "ndcg10_test": round(float(nd[n_val:].mean()), 4),
                             "ndcg10_todos": round(float(nd.mean()), 4),
                             "hit10_todos": media(df, "hit10")})
    M = np.vstack(nd_casos)
    k_best = max(range(len(grid)), key=lambda i: (grid[i]["ndcg10_todos"], grid[i]["hit10_todos"]))
    mejor = grid[k_best]
    if args.auto:
        wb_blk, esq_mejor, wb_b = claves[k_best]
        umbral_fijo = None
    else:
        wb_blk, esq_mejor, wb_b = CONFIG_FINAL["pesos_bloques"], CONFIG_FINAL["esquema"], CONFIG_FINAL["peso_busqueda"]
        umbral_fijo = CONFIG_FINAL["umbral"]
        k_best = claves.index((wb_blk, esq_mejor, wb_b))
    res["seleccion_configuracion"] = {
        "criterio": "NDCG@10 sobre A_val + A_test juntos",
        "elegida_por_maximo": mejor,
        "config_final": {"pesos_bloques": list(wb_blk), "esquema": esq_mejor, "peso_busqueda": wb_b,
                         "umbral": umbral_fijo, "fuente": "--auto (máximo)" if args.auto else "CONFIG_FINAL (fijada)",
                         "grid_de_la_final": grid[k_best]},
        "top5": sorted(grid, key=lambda g: (-g["ndcg10_todos"], -g["hit10_todos"]))[:5],
        "grid": grid,
    }
    print("Máximo del grid:", mejor)
    print("Configuración final:", res["seleccion_configuracion"]["config_final"])

    # estabilidad: elegir con la mitad de los usuarios, medir en la otra mitad
    usr = np.array([c["usuario"] for c in casos_all])
    uu = np.unique(usr)
    sin_b = [i for i, k in enumerate(claves) if k[2] == 0]
    sel, fija, sinb, con_b, usa_texto = [], [], [], [], []
    for _ in range(args.splits):
        mitad = set(rng.choice(uu, len(uu) // 2, replace=False).tolist())
        a = np.array([u in mitad for u in usr])
        k = int(M[:, a].mean(axis=1).argmax())
        ks = sin_b[int(M[sin_b][:, a].mean(axis=1).argmax())]
        sel.append(M[k, ~a].mean())
        fija.append(M[k_best, ~a].mean())
        sinb.append(M[ks, ~a].mean())
        con_b.append(claves[k][2] > 0)
        usa_texto.append(claves[k][0][0] > 0)
    res["estabilidad_seleccion"] = {
        "repeticiones": args.splits,
        "ndcg10_otra_mitad_eligiendo_con_busqueda": round(float(np.mean(sel)), 4),
        "ndcg10_otra_mitad_eligiendo_sin_busqueda": None,
        "dif_con_vs_sin_busqueda": round(float(np.mean(np.array(sel) - np.array(sinb))), 4),
        "pct_repeticiones_donde_con_busqueda_supera": round(float(np.mean(np.array(sel) > np.array(sinb))), 3),
        "pct_repeticiones_que_eligen_w_busqueda_positivo": round(float(np.mean(con_b)), 3),
        "pct_repeticiones_que_eligen_texto_positivo": round(float(np.mean(usa_texto)), 3),
        "nota": "mide el procedimiento de selección en datos no vistos por esa selección (mitades de usuarios)",
    }
    res["estabilidad_seleccion"]["ndcg10_otra_mitad_eligiendo_sin_busqueda"] = round(float(np.mean(sinb)), 4)
    print("Estabilidad:", res["estabilidad_seleccion"])

    X = armar_X(T, C, P, wb_blk)
    Q = bq.matriz_Q(Tq, Cq, wb_blk, P.shape[1]) if busq is not None else None

    # --- 2) evaluación final por protocolo ---
    modelos = {
        "random": (score_random, {}),
        "popularidad": (score_popularidad, {}),
        "colaborativo_itemknn": (score_knn, {"esquema": esq_mejor}),
        "contenido_sin_busqueda": (bq.score_contenido_busq, {"X": X, "Q": Q, "wb": 0, "esquema": esq_mejor}),
        "contenido": (bq.score_contenido_busq, {"X": X, "Q": Q, "wb": wb_b, "esquema": esq_mejor}),
    }
    tabla, por_caso = [], {}
    res["resultados"] = {}
    for proto, cs in casos_final.items():
        por_caso[proto] = {}
        res["resultados"][proto] = {}
        for nombre, (fn, kw) in modelos.items():
            df = evaluar(cs, fn, ctx, eleg_rows, carga, rng, **kw)
            por_caso[proto][nombre] = df
            r = resumir(df, len(eleg_rows), cola_rows, rng)
            res["resultados"][proto][nombre] = r
            tabla.append({"protocolo": proto, "segmento": "todos", "modelo": nombre, **r})
    res["notas_protocolos"] = {
        "A_todos": "A_val + A_test juntos (cada usuario aporta hasta 2 casos); la configuración del modelo de "
                   "contenido se eligió sobre estos mismos casos, así que sus números son optimistas; los "
                   "intervalos tratan los casos como independientes",
        "B": "no participó de ninguna selección; muestra chica",
    }

    # --- 3) umbral de personalización (híbrido) ---
    pc = por_caso["A_todos"]
    curva = []
    for u in UMBRALES:
        h = combinar(pc["contenido"], pc["popularidad"], u)
        curva.append({"umbral": u, "hit10": media(h, "hit10"), "ndcg10": media(h, "ndcg10"),
                      "pct_casos_personalizados": round(float((pc["contenido"]["n_prof"] >= u).mean()), 4)})
    umbral = max(curva, key=lambda c: (c["ndcg10"], -c["umbral"]))["umbral"]
    res["umbral"] = {"curva": curva, "maximo_en_curva": umbral,
                     "nota": "con pocos casos es orientativo; umbral 1 vs 2 difieren en ~0.001 de NDCG"}
    if umbral_fijo is not None:
        umbral = umbral_fijo
    res["umbral"]["elegido"] = umbral
    print("Umbral elegido:", umbral)
    for proto in casos_final:
        h = combinar(por_caso[proto]["contenido"], por_caso[proto]["popularidad"], umbral)
        por_caso[proto]["hibrido"] = h
        r = resumir(h, len(eleg_rows), cola_rows, rng)
        res["resultados"][proto]["hibrido"] = r
        tabla.append({"protocolo": proto, "segmento": "todos", "modelo": "hibrido", **r})

    # --- 4) por segmento (según historial) en A_todos ---
    res["segmentos_A_todos"] = {}
    for nombre_seg, lo, hi in SEGMENTOS:
        res["segmentos_A_todos"][nombre_seg] = {}
        for nombre in ["random", "popularidad", "colaborativo_itemknn", "contenido"]:
            df = por_caso["A_todos"][nombre]
            k = ((df["n_prof"] >= lo) & (df["n_prof"] <= hi)).tolist()
            sub = df[k].copy()
            if len(sub) == 0:
                continue
            sub.attrs["top10"] = [t for t, kk in zip(df.attrs["top10"], k) if kk]
            r = resumir(sub, len(eleg_rows), cola_rows, rng, con_ic=False)
            res["segmentos_A_todos"][nombre_seg][nombre] = r
            tabla.append({"protocolo": "A_todos", "segmento": f"historial {nombre_seg}", "modelo": nombre, **r})

    # --- 5) diferencias pareadas con intervalo de confianza ---
    res["diferencias"] = {}
    for proto in casos_final:
        p = por_caso[proto]
        d = {}
        for metrica in ("hit10", "ndcg10"):
            d[metrica] = {
                "contenido_vs_random": dif_pareada(p["contenido"][metrica], p["random"][metrica], rng),
                "contenido_vs_popularidad": dif_pareada(p["contenido"][metrica], p["popularidad"][metrica], rng),
                "contenido_vs_colaborativo": dif_pareada(p["contenido"][metrica], p["colaborativo_itemknn"][metrica], rng),
                "contenido_vs_sin_busqueda": dif_pareada(p["contenido"][metrica], p["contenido_sin_busqueda"][metrica], rng),
                "hibrido_vs_popularidad": dif_pareada(p["hibrido"][metrica], p["popularidad"][metrica], rng),
            }
        res["diferencias"][proto] = d
    res["forzados_target_fuera_de_candidatos"] = {p: por_caso[p]["contenido"].attrs["forzados"] for p in por_caso}

    # --- salidas ---
    pd.DataFrame(tabla).to_csv(dout / "resultados_tabla.csv", index=False)
    with open(dout / "resultados_evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    orden = ["random", "popularidad", "colaborativo_itemknn", "contenido", "hibrido"]
    colores = ["#9ca3af", "#f59e0b", "#10b981", "#2563eb", "#7c3aed"]
    n_casos = {"A_todos": len(casos_all), "B": len(casos["B"])}
    for ax, proto, titulo in zip(axes, ("A_todos", "B"),
                                 ("Último y penúltimo producto nuevo", "Último producto contactado")):
        vals = [res["resultados"][proto][m]["hit10"] for m in orden]
        err = [[v - res["resultados"][proto][m]["hit10_ic95"][0] for v, m in zip(vals, orden)],
               [res["resultados"][proto][m]["hit10_ic95"][1] - v for v, m in zip(vals, orden)]]
        ax.bar(range(len(orden)), vals, yerr=err, color=colores, capsize=3)
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels(["random", "popul.", "colab.", "contenido", "híbrido"], fontsize=9)
        ax.set_title(f"{titulo} (n={n_casos[proto]})", fontsize=10)
        ax.set_ylabel("Hit@10")
    fig.tight_layout()
    fig.savefig(dout / "comparacion_modelos.png", dpi=150)
    plt.close(fig)

    # --- modelo final con todos los datos (para la API) ---
    guardar_modelo(dout / "modelo.joblib", prod, ctx, X, tfidf, cats, nombre_a_id, P.shape[1],
                   wb_blk, esq_mejor, umbral, wb_b)

    print("\n===== RESULTADOS =====")
    print(pd.DataFrame(tabla)[["protocolo", "segmento", "modelo", "n_casos", "hit5", "hit10", "ndcg10", "mrr",
                                 "cobertura10", "pct_cola_top10"]].to_string(index=False))
    print("\nListo. Archivos en:", dout.resolve())


if __name__ == "__main__":
    main()
