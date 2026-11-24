import csv
import json
import shutil

import pytest

from mft import snapshot as sn
from tests.snapshot_sintetico import crear_snapshot


@pytest.fixture()
def snap(tmp_path):
    return crear_snapshot(tmp_path / "2026-11-01")


def test_snapshot_valido(snap):
    v = sn.validar(snap)
    assert v.ok and v.filas["producto"] == 30 and not v.avisos


def test_sin_manifest_es_incompleto(snap):
    (snap / "manifest.json").unlink()
    assert "falta manifest.json (snapshot incompleto)" in sn.validar(snap).errores


def test_falta_una_tabla(snap):
    (snap / "favorito.csv").unlink()
    assert any("falta la tabla favorito" in e for e in sn.validar(snap).errores)


def test_filas_que_no_coinciden_con_el_manifest(snap):
    with open(snap / "consulta.csv", "a", newline="") as f:
        csv.writer(f).writerow(["2026-10-01 10:00:00", "", 1, 1])
    assert any("consulta" in e and "filas" in e for e in sn.validar(snap).errores)


def test_hash_distinto_detecta_archivo_corrupto(snap):
    m = json.loads((snap / "manifest.json").read_text())
    m["tablas"]["producto"]["sha256"] = "0" * 64
    (snap / "manifest.json").write_text(json.dumps(m))
    assert any("sha256" in e for e in sn.validar(snap).errores)


def test_columnas_personales_se_rechazan(snap):
    with open(snap / "usuario.csv", newline="") as f:
        filas = list(csv.reader(f))
    filas[0].append("email")
    for r in filas[1:]:
        r.append("a@b.com")
    with open(snap / "usuario.csv", "w", newline="") as f:
        csv.writer(f).writerows(filas)
    sn.escribir_manifest(snap)
    assert any("prohibidas" in e and "email" in e for e in sn.validar(snap).errores)


def test_faltan_columnas_requeridas(snap):
    with open(snap / "favorito.csv", newline="") as f:
        filas = list(csv.reader(f))
    with open(snap / "favorito.csv", "w", newline="") as f:
        csv.writer(f).writerows([[c for c in r[:2]] for r in filas])
    sn.escribir_manifest(snap)
    assert any("favorito" in e and "faltan columnas" in e for e in sn.validar(snap).errores)


def test_columnas_extra_inofensivas_se_descartan(snap, tmp_path):
    with open(snap / "categoria.csv", newline="") as f:
        filas = list(csv.reader(f))
    filas[0].append("icono")
    for r in filas[1:]:
        r.append("x.png")
    with open(snap / "categoria.csv", "w", newline="") as f:
        csv.writer(f).writerows(filas)
    sn.escribir_manifest(snap)
    v = sn.validar(snap)
    assert v.ok and any("icono" in a for a in v.avisos)
    sn.reducir_al_contrato(snap, tmp_path / "red")
    assert sn.encabezado(tmp_path / "red" / "categoria.csv") == ["id", "nombre"]


def test_formato_desconocido(snap):
    m = json.loads((snap / "manifest.json").read_text())
    m["formato"] = 99
    (snap / "manifest.json").write_text(json.dumps(m))
    assert any("formato" in e for e in sn.validar(snap).errores)
