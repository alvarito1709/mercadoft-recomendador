"""
Contrato del snapshot que el backend deja en el bucket cada noche (documentado en docs/contrato_snapshot.md).

Un snapshot es una carpeta `AAAA-MM-DD/` con un CSV por tabla y un `manifest.json` que se escribe AL FINAL:
si el manifest existe, el snapshot está completo. Este módulo define las tablas/columnas permitidas, valida un
snapshot y lo reduce a las columnas del contrato (defensa en profundidad: aunque el backend mandara de más, esas
columnas nunca llegan al pipeline).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

FORMATO = 1
NOMBRE_SNAPSHOT = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# tabla -> columnas requeridas (el CSV se llama <tabla>.csv; encabezado en minúsculas)
CONTRATO: dict[str, list[str]] = {
    "producto": ["id", "nombre", "descripcion", "precio", "desactivada", "estado_revision", "fecha_carga", "emprendedor_id"],
    "producto_categoria": ["producto_id", "categoria_id"],
    "categoria": ["id", "nombre"],
    "emprendedor": ["id", "usuario_id", "estado_aprobacion"],
    "producto_borrado": ["producto_original_id", "nombre", "descripcion", "precio", "emprendedor_id", "fecha_carga"],
    "usuario": ["id", "eliminado", "email_verificado", "rol_id", "fecha_de_registro"],
    "producto_visto": ["fecha", "producto_original_id", "producto_id", "usuario_id"],
    "favorito": ["fecha_creado", "producto_id", "usuario_id"],
    "consulta": ["fecha", "producto_original_id", "producto_id", "usuario_id"],
    "calificacion": ["calificacion", "fecha", "producto_original_id", "producto_id", "usuario_id"],
    "compartido": ["fecha", "tipo", "producto_original_id", "producto_id", "usuario_id"],
    "busqueda": ["fecha_busqueda", "termino_busqueda", "usuario_id"],
}

# Datos personales / texto libre que NUNCA deben estar en un snapshot. Si aparecen, el snapshot se rechaza.
PROHIBIDAS = {
    "email", "password", "documento", "numero_celular", "nombres", "apellidos", "username", "comentario",
    "motivo_rechazo", "facebook", "instagram", "imagen_perfil", "nombre_imagen_perfil", "anonimo_id",
}


@dataclass
class Validacion:
    errores: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    filas: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errores


def _abrir(path: Path):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    return open(path, newline="", encoding="utf-8-sig")


def encabezado(path: Path) -> list[str]:
    with _abrir(path) as f:
        fila = next(csv.reader(f), [])
    return [c.strip().lower() for c in fila]


def contar_filas(path: Path) -> int:
    with _abrir(path) as f:
        r = csv.reader(f)
        next(r, None)
        return sum(1 for _ in r)


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def parsear_fecha(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def validar(carpeta: Path) -> Validacion:
    """Revisa manifest, presencia de tablas, columnas, cantidad de filas, hashes y columnas prohibidas."""
    v = Validacion()
    carpeta = Path(carpeta)
    mf = carpeta / "manifest.json"
    if not mf.exists():
        v.errores.append("falta manifest.json (snapshot incompleto)")
        return v
    try:
        v.manifest = json.loads(mf.read_text(encoding="utf-8"))
    except Exception as e:
        v.errores.append(f"manifest.json ilegible: {e}")
        return v
    if v.manifest.get("formato") != FORMATO:
        v.errores.append(f"formato de manifest {v.manifest.get('formato')!r}, se espera {FORMATO}")
    try:
        parsear_fecha(v.manifest.get("generado"))
    except Exception:
        v.errores.append("manifest.generado ausente o no es una fecha ISO-8601")
    tablas = v.manifest.get("tablas", {})
    for t, cols in CONTRATO.items():
        p = carpeta / f"{t}.csv"
        if t not in tablas or not p.exists():
            v.errores.append(f"falta la tabla {t}")
            continue
        enc = encabezado(p)
        prohibidas = sorted(set(enc) & PROHIBIDAS)
        if prohibidas:
            v.errores.append(f"{t}: contiene columnas prohibidas (datos personales): {prohibidas}")
        faltan = [c for c in cols if c not in enc]
        if faltan:
            v.errores.append(f"{t}: faltan columnas {faltan}")
        extra = [c for c in enc if c not in cols and c not in PROHIBIDAS]
        if extra:
            v.avisos.append(f"{t}: columnas fuera del contrato (se descartan): {extra}")
        n = contar_filas(p)
        v.filas[t] = n
        esperado = tablas[t].get("filas")
        if esperado is not None and int(esperado) != n:
            v.errores.append(f"{t}: el manifest dice {esperado} filas y el archivo tiene {n}")
        h = tablas[t].get("sha256")
        if h and sha256_archivo(p) != h:
            v.errores.append(f"{t}: sha256 no coincide (archivo incompleto o corrupto)")
    if v.filas.get("producto", 0) == 0:
        v.errores.append("la tabla producto está vacía")
    return v


def reducir_al_contrato(origen: Path, destino: Path) -> None:
    """Copia cada tabla dejando solo las columnas del contrato, en el orden del contrato."""
    destino.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    for t, cols in CONTRATO.items():
        with _abrir(Path(origen) / f"{t}.csv") as fi, open(destino / f"{t}.csv", "w", newline="", encoding="utf-8") as fo:
            r = csv.reader(fi)
            enc = [c.strip().lower() for c in next(r)]
            idx = [enc.index(c) for c in cols]
            w = csv.writer(fo)
            w.writerow(cols)
            for fila in r:
                w.writerow([fila[i] if i < len(fila) else "" for i in idx])


def escribir_manifest(carpeta: Path, generado: str | None = None) -> dict:
    """Escribe manifest.json para los CSV de la carpeta. Referencia para el backend; también lo usan los tests."""
    carpeta = Path(carpeta)
    m = {"formato": FORMATO, "generado": generado or datetime.now().astimezone().isoformat(timespec="seconds"),
         "tablas": {}}
    for t in CONTRATO:
        p = carpeta / f"{t}.csv"
        if p.exists():
            m["tablas"][t] = {"archivo": p.name, "filas": contar_filas(p), "sha256": sha256_archivo(p)}
    (carpeta / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m
