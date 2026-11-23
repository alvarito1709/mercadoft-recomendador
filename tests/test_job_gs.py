"""Prueba la ruta gs:// del job con un `gcloud` falso que emula `storage ls` y `storage cp` sobre una carpeta local."""
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mft import job_nocturno as job
from tests.snapshot_sintetico import crear_snapshot

FAKE = '''#!{py}
import os, shutil, sys
from pathlib import Path
raiz = Path(os.environ["FAKE_GCS_ROOT"])
a = sys.argv[1:]
assert a[0] == "storage", a
def local(url):
    assert url.startswith("gs://"), url
    return raiz / url[len("gs://"):]
if a[1] == "ls":
    url = a[2]
    p = local(url.rstrip("/"))
    if url.endswith("/"):
        if not p.is_dir():
            print("One or more URLs matched no objects.", file=sys.stderr); sys.exit(1)
        for d in sorted(p.iterdir()):
            if d.is_dir():
                print(url + d.name + "/")
        sys.exit(0)
    if p.exists():
        print(url); sys.exit(0)
    print("One or more URLs matched no objects.", file=sys.stderr); sys.exit(1)
if a[1] == "cp":
    assert "--recursive" in a
    src, dst = a[-2], Path(a[-1])
    assert src.endswith("/*"), src
    origen = local(src[:-2])
    for f in origen.iterdir():
        shutil.copy2(f, dst / f.name)
    sys.exit(0)
sys.exit(2)
'''


@pytest.fixture()
def gs(tmp_path, monkeypatch):
    fake = tmp_path / "gcloud"
    fake.write_text(FAKE.format(py=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    root = tmp_path / "gcs"
    monkeypatch.setenv("FAKE_GCS_ROOT", str(root))
    crear_snapshot(root / "bucket-reco" / "snapshots" / "2026-11-01", generado="2026-11-01T03:30:00-03:00")
    (root / "bucket-reco" / "snapshots" / "2026-11-02").mkdir()          # export sin manifest: se ignora
    return job.Config(origen="gs://bucket-reco/snapshots", base=tmp_path / "base", gcloud=str(fake)), root


AHORA = datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc)


def test_listado_y_manifest_via_gcloud(gs):
    cfg, _ = gs
    assert job.listar_snapshots(cfg) == ["2026-11-01", "2026-11-02"]
    assert job.tiene_manifest(cfg, "2026-11-01") and not job.tiene_manifest(cfg, "2026-11-02")


def test_job_completo_desde_gs(gs):
    cfg, _ = gs
    assert job.ejecutar(cfg, AHORA) == "publicado"
    assert (cfg.base / "models" / "current" / "modelo.joblib").exists()
    assert job.ejecutar(cfg, AHORA) == "sin_novedades"


def test_error_de_gcloud_no_toca_nada(gs):
    cfg, root = gs
    shutil.rmtree(root / "bucket-reco")
    with pytest.raises(job.ErrorJob):
        job.ejecutar(cfg, AHORA)
    assert not (cfg.base / "models" / "current" / "modelo.joblib").exists()
