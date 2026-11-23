from datetime import datetime, timezone
from pathlib import Path

import joblib
import pytest

from mft import job_nocturno as job
from mft.recomendador import Modelo
from tests.snapshot_sintetico import crear_snapshot

AHORA = datetime(2026, 11, 2, 4, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("job")
    origen = raiz / "snapshots"
    crear_snapshot(origen / "2026-11-01", generado="2026-11-01T03:30:00-03:00")
    cfg = job.Config(origen=str(origen), base=raiz / "base")
    return cfg, origen


def version_actual(cfg):
    return job.leer_estado(cfg.base)["actual"]["snapshot"]


def modelo_actual(cfg):
    return Modelo(joblib.load(cfg.base / "models" / "current" / "modelo.joblib"))


def test_1_primer_run_publica_un_modelo_usable(entorno):
    cfg, _ = entorno
    assert job.ejecutar(cfg, AHORA) == "publicado"
    m = modelo_actual(cfg)
    assert len(m.ids) == 30 and int(m.elegible.sum()) == 29        # 1 desactivado
    assert m.recomendar([(1, "vista")], ["joyería"], 5).modo == "contenido"
    est = job.leer_estado(cfg.base)["actual"]
    assert est["snapshot"] == "2026-11-01" and est["modo"] == "entrenar"
    assert not (cfg.base / "descargas").exists()                     # el crudo del bucket no se conserva


def test_2_sin_novedades_no_toca_el_modelo(entorno):
    cfg, _ = entorno
    ruta = cfg.base / "models" / "current" / "modelo.joblib"
    antes = ruta.stat().st_mtime_ns
    assert job.ejecutar(cfg, AHORA) == "sin_novedades"
    assert ruta.stat().st_mtime_ns == antes


def test_3_snapshot_sin_manifest_se_ignora(entorno):
    cfg, origen = entorno
    (origen / "2026-11-02").mkdir()
    (origen / "2026-11-02" / "producto.csv").write_text("id\n")      # export a medio escribir
    assert job.ejecutar(cfg, AHORA) == "sin_novedades"
    assert version_actual(cfg) == "2026-11-01"


def test_4_snapshot_invalido_no_toca_el_modelo(entorno):
    cfg, origen = entorno
    malo = crear_snapshot(origen / "2026-11-03", generado="2026-11-03T03:30:00-03:00", semilla=2)
    (malo / "favorito.csv").unlink()
    with pytest.raises(job.ErrorJob, match="inválido"):
        job.ejecutar(cfg, AHORA)
    assert version_actual(cfg) == "2026-11-01"
    assert modelo_actual(cfg).recomendar([], [], 3).items


def test_5_caida_fuerte_de_eventos_se_frena(entorno):
    cfg, origen = entorno
    import shutil
    shutil.rmtree(origen / "2026-11-03")
    crear_snapshot(origen / "2026-11-04", generado="2026-11-04T03:30:00-03:00", semilla=3, factor_eventos=0.1)
    with pytest.raises(job.ErrorJob, match="cayeron"):
        job.ejecutar(cfg, AHORA)
    assert version_actual(cfg) == "2026-11-01"


def test_6_snapshot_nuevo_valido_se_publica_y_se_puede_volver_atras(entorno):
    cfg, origen = entorno
    import shutil
    shutil.rmtree(origen / "2026-11-04")
    crear_snapshot(origen / "2026-11-05", generado="2026-11-05T03:30:00-03:00", semilla=4)
    assert job.ejecutar(cfg, AHORA) == "publicado"
    assert version_actual(cfg) == "2026-11-05"
    assert job.rollback(cfg) == "rollback"
    assert version_actual(cfg) == "2026-11-01"
    with pytest.raises(job.ErrorJob):
        job.rollback(cfg, "2020-01-01")


def test_7_el_rollback_se_respeta_hasta_que_haya_un_snapshot_nuevo(entorno):
    cfg, origen = entorno
    # tras el rollback sirve 2026-11-01, pero el último procesado fue 2026-11-05: no se vuelve a publicar solo
    assert version_actual(cfg) == "2026-11-01"
    assert job.ejecutar(cfg, AHORA) == "sin_novedades"
    assert version_actual(cfg) == "2026-11-01"
    # con --forzar sí se republica el más nuevo
    forzado = job.Config(origen=cfg.origen, base=cfg.base, forzar=True)
    assert job.ejecutar(forzado, AHORA) == "publicado"
    assert version_actual(cfg) == "2026-11-05"


def test_7b_snapshot_desactualizado_alerta(entorno):
    cfg, _ = entorno
    tarde = datetime(2026, 11, 20, tzinfo=timezone.utc)
    with pytest.raises(job.ErrorJob) as e:
        job.ejecutar(cfg, tarde)
    assert e.value.codigo == 2


def test_8_no_corren_dos_a_la_vez(tmp_path):
    with job.Bloqueo(tmp_path):
        with pytest.raises(job.ErrorJob) as e:
            with job.Bloqueo(tmp_path):
                pass
    assert e.value.codigo == 3
    with job.Bloqueo(tmp_path):      # se libera al salir
        pass


def test_9_retencion_de_versiones(tmp_path):
    base = tmp_path
    vers = base / "models" / "versiones"
    for n in ("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"):
        (vers / n).mkdir(parents=True)
        (vers / n / "modelo.joblib").write_bytes(b"x")
    job.limpiar(base, job.Config("x", base, conservar=2), {"actual": {"snapshot": "2026-10-01"}})
    assert sorted(p.name for p in vers.iterdir()) == ["2026-10-01", "2026-10-03", "2026-10-04"]   # conserva las 2 últimas + la vigente
