"""
Job nocturno de la VM del recomendador: baja el último snapshot del bucket, lo valida, entrena y publica el modelo.

    python -m mft.job_nocturno --origen gs://<bucket>/snapshots --base /var/lib/mft
    python -m mft.job_nocturno --origen /ruta/local/snapshots --base /tmp/mft      # pruebas
    python -m mft.job_nocturno --base /var/lib/mft --rollback [AAAA-MM-DD]          # volver a una versión anterior

Garantía principal: si cualquier paso falla, el modelo que está sirviendo la API NO se toca.

Estructura bajo --base:
  models/current/modelo.joblib     lo que lee la API (se reemplaza con os.replace: rename atómico)
  models/versiones/<snapshot>/     últimas N versiones, para volver atrás
  trabajo/<snapshot>/{data,out}    datos reducidos al contrato y salidas del pipeline (se conservan las últimas 2)
  informes/<snapshot>/             resultados de la evaluación (solo con --modo evaluar)
  estado.json  .salt  logs/job.log  .lock

Códigos de salida: 0 ok o sin novedades · 1 error (modelo anterior intacto) · 2 snapshot desactualizado · 3 ya hay otra corrida.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import snapshot as sn
from .recomendador import Modelo

log = logging.getLogger("mft.job")

REPO = Path(__file__).resolve().parents[2]
EDAD_MAXIMA_SNAPSHOT = timedelta(hours=48)
CAIDA_MAXIMA_EVENTOS = 0.30       # si los eventos caen más que esto respecto de la versión publicada, se frena
CAIDA_MAXIMA_ELEGIBLES = 0.50
LOCK_VENCIDO_S = 6 * 3600


class ErrorJob(Exception):
    def __init__(self, mensaje, codigo=1):
        super().__init__(mensaje)
        self.codigo = codigo


@dataclass
class Config:
    origen: str
    base: Path
    modo: str = "entrenar"            # "entrenar" (sin evaluar, segundos) | "evaluar" (corrida completa, minutos)
    conservar: int = 7
    forzar: bool = False
    gcloud: str = "gcloud"


# ----------------------------------------------------------------------------- estado y archivos
def _json_atomico(ruta: Path, obj) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ruta)


def leer_estado(base: Path) -> dict:
    p = base / "estado.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"actual": None, "historial": []}


def _salt(base: Path) -> str:
    p = base / ".salt"
    if not p.exists():
        base.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(16))
        try:
            p.chmod(0o600)
        except OSError:
            pass
    return p.read_text().strip()


class Bloqueo:
    """Evita dos corridas a la vez. Archivo exclusivo; se considera abandonado pasadas LOCK_VENCIDO_S."""

    def __init__(self, base: Path):
        self.ruta = base / ".lock"

    def __enter__(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if self.ruta.exists() and time.time() - self.ruta.stat().st_mtime > LOCK_VENCIDO_S:
            log.warning("bloqueo abandonado, se ignora")
            self.ruta.unlink()
        try:
            fd = os.open(self.ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ErrorJob("ya hay otra corrida en curso", 3)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *a):
        self.ruta.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- origen (bucket o carpeta local)
def _es_gs(origen: str) -> bool:
    return origen.startswith("gs://")


def _gcloud(cfg: Config, *args, check=True) -> subprocess.CompletedProcess:
    cmd = [cfg.gcloud, "storage", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise ErrorJob(f"falló `{' '.join(cmd)}`: {r.stderr.strip()[:500]}")
    return r


def listar_snapshots(cfg: Config) -> list[str]:
    """Nombres AAAA-MM-DD presentes en el origen, ordenados de más viejo a más nuevo."""
    if _es_gs(cfg.origen):
        r = _gcloud(cfg, "ls", cfg.origen.rstrip("/") + "/")
        nombres = [l.rstrip("/").split("/")[-1] for l in r.stdout.splitlines() if l.strip().endswith("/")]
    else:
        raiz = Path(cfg.origen)
        nombres = [p.name for p in raiz.iterdir() if p.is_dir()] if raiz.exists() else []
    return sorted(n for n in nombres if sn.NOMBRE_SNAPSHOT.match(n))


def tiene_manifest(cfg: Config, nombre: str) -> bool:
    if _es_gs(cfg.origen):
        return _gcloud(cfg, "ls", f"{cfg.origen.rstrip('/')}/{nombre}/manifest.json", check=False).returncode == 0
    return (Path(cfg.origen) / nombre / "manifest.json").exists()


def traer(cfg: Config, nombre: str, destino: Path) -> None:
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    if _es_gs(cfg.origen):
        _gcloud(cfg, "cp", "--recursive", f"{cfg.origen.rstrip('/')}/{nombre}/*", str(destino))
    else:
        shutil.copytree(Path(cfg.origen) / nombre, destino, dirs_exist_ok=True)


# ----------------------------------------------------------------------------- pipeline
def _correr(cmd: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        cola = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
        raise ErrorJob(f"falló {Path(cmd[1]).name} (ver {log_path}): {cola}")


def verificar_modelo(ruta: Path, estado: dict, eventos: int | None, forzar: bool) -> Modelo:
    """Pruebas de humo del modelo candidato antes de publicarlo."""
    m = Modelo(__import__("joblib").load(ruta))
    ant = estado.get("actual") or {}
    elegibles = int(m.elegible.sum())
    if elegibles == 0:
        raise ErrorJob("el modelo candidato no tiene productos elegibles")
    if not forzar:
        if ant.get("elegibles") and elegibles < ant["elegibles"] * (1 - CAIDA_MAXIMA_ELEGIBLES):
            raise ErrorJob(f"los productos elegibles cayeron de {ant['elegibles']} a {elegibles}; revisar el snapshot (usar --forzar si es correcto)")
        if ant.get("eventos") and eventos is not None and eventos < ant["eventos"] * (1 - CAIDA_MAXIMA_EVENTOS):
            raise ErrorJob(f"los eventos cayeron de {ant['eventos']} a {eventos}; revisar el snapshot (usar --forzar si es correcto)")
    r = m.recomendar([], [], n=10)
    no_eleg = {int(m.ids[i]) for i in range(len(m.ids)) if not m.elegible[i]}
    if len(r.items) == 0 or any(p in no_eleg for p, _ in r.items):
        raise ErrorJob("prueba de humo fallida: recomendación en frío vacía o con productos no elegibles")
    top = r.items[0][0]
    r2 = m.recomendar([(top, "vista")], [], n=10)
    if r2.modo != "contenido" or not r2.items or any(p == top or p in no_eleg for p, _ in r2.items):
        raise ErrorJob("prueba de humo fallida: la recomendación con historial no es válida")
    return m


def publicar(base: Path, nombre: str, modelo_ruta: Path) -> None:
    """Copia a versiones/ y reemplaza models/current/ con un rename atómico."""
    ver = base / "models" / "versiones" / nombre
    ver.mkdir(parents=True, exist_ok=True)
    if modelo_ruta.resolve() != (ver / "modelo.joblib").resolve():
        shutil.copy2(modelo_ruta, ver / "modelo.joblib")
    actual = base / "models" / "current"
    actual.mkdir(parents=True, exist_ok=True)
    tmp = actual / "modelo.joblib.tmp"
    shutil.copy2(ver / "modelo.joblib", tmp)
    os.replace(tmp, actual / "modelo.joblib")      # el rename es atómico: la API nunca ve un archivo a medias


def limpiar(base: Path, cfg: Config, estado: dict) -> None:
    vers = base / "models" / "versiones"
    if vers.exists():
        todas = sorted(p.name for p in vers.iterdir() if p.is_dir())
        vigente = (estado.get("actual") or {}).get("snapshot")
        for n in todas[:-cfg.conservar]:
            if n != vigente:
                shutil.rmtree(vers / n, ignore_errors=True)
    tr = base / "trabajo"
    if tr.exists():
        for n in sorted(p.name for p in tr.iterdir() if p.is_dir())[:-2]:
            shutil.rmtree(tr / n, ignore_errors=True)
    shutil.rmtree(base / "descargas", ignore_errors=True)    # el crudo del bucket no se conserva


def ejecutar(cfg: Config, ahora: datetime | None = None) -> str:
    """Devuelve "publicado" o "sin_novedades". Lanza ErrorJob si algo falla (modelo anterior intacto)."""
    ahora = ahora or datetime.now().astimezone()
    base = cfg.base
    estado = leer_estado(base)

    # 1) último snapshot COMPLETO (con manifest)
    candidatos = [n for n in reversed(listar_snapshots(cfg)) if tiene_manifest(cfg, n)]
    if not candidatos:
        raise ErrorJob(f"no hay ningún snapshot completo en {cfg.origen}")
    nombre = candidatos[0]
    desc = base / "descargas" / nombre
    traer(cfg, nombre, desc)
    v = sn.validar(desc)
    for a in v.avisos:
        log.warning("snapshot %s: %s", nombre, a)
    if not v.ok:
        raise ErrorJob(f"snapshot {nombre} inválido: " + "; ".join(v.errores))
    generado = v.manifest["generado"]
    gen_dt = sn.parsear_fecha(generado)
    if gen_dt.tzinfo is None:
        gen_dt = gen_dt.astimezone()
    # "ultimo_procesado" es independiente de lo que sirve la API: un rollback cambia lo publicado pero NO debe
    # hacer que la noche siguiente se vuelva a publicar el mismo snapshot que se descartó.
    procesado = estado.get("ultimo_procesado") or estado.get("actual") or {}

    # 2) ¿hay novedades?
    if not cfg.forzar and procesado.get("snapshot") == nombre and procesado.get("generado") == generado:
        edad = ahora - gen_dt
        if edad > EDAD_MAXIMA_SNAPSHOT:
            raise ErrorJob(f"sin snapshot nuevo hace {edad}: el export del backend no está corriendo", 2)
        log.info("sin novedades: el snapshot %s ya está publicado", nombre)
        return "sin_novedades"
    if ahora - gen_dt > EDAD_MAXIMA_SNAPSHOT:
        log.warning("el snapshot %s tiene %s de antigüedad", nombre, ahora - gen_dt)

    # 3) reducir al contrato y correr el pipeline
    trabajo = base / "trabajo" / nombre
    shutil.rmtree(trabajo, ignore_errors=True)
    data, out = trabajo / "data", trabajo / "out"
    sn.reducir_al_contrato(desc, data)
    out.mkdir(parents=True)
    env = {**os.environ, "RECO_SALT": _salt(base)}
    logs = base / "logs"
    py = sys.executable
    _correr([py, str(REPO / "eda_recomendador.py"), "--data-dir", str(data), "--out", str(out)], logs / f"{nombre}-eda.log", env)
    resumen = json.loads((out / "eda_resumen.json").read_text(encoding="utf-8"))
    eventos = resumen.get("eventos", {}).get("total")
    args = [py, str(REPO / "modelo_eval.py"), "--in", str(out), "--out", str(out), "--data-dir", str(data)]
    if cfg.modo == "entrenar":
        args.append("--solo-entrenar")
    _correr(args, logs / f"{nombre}-modelo.log", env)

    # 4) verificar y publicar
    candidato = out / "modelo.joblib"
    m = verificar_modelo(candidato, estado, eventos, cfg.forzar)
    publicar(base, nombre, candidato)
    if cfg.modo == "evaluar":
        inf = base / "informes" / nombre
        inf.mkdir(parents=True, exist_ok=True)
        for f in ("resultados_evaluacion.json", "resultados_tabla.csv", "comparacion_modelos.png"):
            if (out / f).exists():
                shutil.copy2(out / f, inf / f)
    estado["actual"] = {"snapshot": nombre, "generado": generado, "modelo_version": m.version, "eventos": eventos,
                        "elegibles": int(m.elegible.sum()), "productos": int(len(m.ids)),
                        "publicado": ahora.isoformat(timespec="seconds"), "modo": cfg.modo}
    estado["ultimo_procesado"] = {"snapshot": nombre, "generado": generado}
    estado["historial"] = (estado.get("historial", []) + [nombre])[-50:]
    _json_atomico(base / "estado.json", estado)
    limpiar(base, cfg, estado)
    log.info("publicado el modelo de %s (%d productos, %d elegibles, %s eventos)", nombre, len(m.ids), int(m.elegible.sum()), eventos)
    return "publicado"


def rollback(cfg: Config, destino: str | None = None, ahora: datetime | None = None) -> str:
    base = cfg.base
    estado = leer_estado(base)
    vers = base / "models" / "versiones"
    disponibles = sorted(p.name for p in vers.iterdir() if (p / "modelo.joblib").exists()) if vers.exists() else []
    vigente = (estado.get("actual") or {}).get("snapshot")
    if destino is None:
        previas = [n for n in disponibles if vigente is None or n < vigente]
        if not previas:
            raise ErrorJob("no hay una versión anterior para volver")
        destino = previas[-1]
    if destino not in disponibles:
        raise ErrorJob(f"la versión {destino} no está disponible (hay: {disponibles})")
    m = Modelo(__import__("joblib").load(vers / destino / "modelo.joblib"))
    publicar(base, destino, vers / destino / "modelo.joblib")
    ahora = ahora or datetime.now().astimezone()
    estado["actual"] = {**(estado.get("actual") or {}), "snapshot": destino, "modelo_version": m.version,
                        "elegibles": int(m.elegible.sum()), "productos": int(len(m.ids)),
                        "publicado": ahora.isoformat(timespec="seconds"), "modo": "rollback"}
    estado["historial"] = (estado.get("historial", []) + [f"rollback:{destino}"])[-50:]
    _json_atomico(base / "estado.json", estado)
    log.warning("rollback: ahora sirve la versión %s", destino)
    return "rollback"


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", default=os.environ.get("RECO_ORIGEN"), help="gs://bucket/carpeta o ruta local con las carpetas AAAA-MM-DD")
    ap.add_argument("--base", default=os.environ.get("RECO_BASE", "/var/lib/mft"))
    ap.add_argument("--modo", choices=["entrenar", "evaluar"], default="entrenar")
    ap.add_argument("--conservar", type=int, default=7)
    ap.add_argument("--forzar", action="store_true", help="publicar aunque las comprobaciones de caída de datos fallen o no haya novedades")
    ap.add_argument("--rollback", nargs="?", const="", metavar="AAAA-MM-DD", help="volver a la versión anterior (o a la indicada)")
    a = ap.parse_args(argv)
    base = Path(a.base)
    base.mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(base / "logs" / "job.log", encoding="utf-8")])
    try:
        with Bloqueo(base):
            if a.rollback is not None:
                rollback(Config(a.origen or "", base, conservar=a.conservar), a.rollback or None)
            else:
                if not a.origen:
                    raise ErrorJob("falta --origen (o RECO_ORIGEN)")
                ejecutar(Config(a.origen, base, a.modo, a.conservar, a.forzar))
        return 0
    except ErrorJob as e:
        log.error("%s", e)
        return e.codigo
    except Exception:
        log.exception("error inesperado; el modelo publicado no se modificó")
        return 1


if __name__ == "__main__":
    sys.exit(main())
