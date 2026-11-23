import os
import time
from pathlib import Path

import joblib
import numpy as np
import pytest

from mft import busqueda as bq
from mft.recomendador import CLAVES_MODELO, Modelo, ModeloStore
from tests.conftest import construir_dict_modelo


def ids_rubro(prod, cid):
    return set(prod.loc[prod["cid"] == cid, "producto_id"])


def test_arranque_en_frio_devuelve_populares_elegibles(modelo, prod):
    r = modelo.recomendar([], [], n=5)
    assert (r.modo, r.motivo) == ("popularidad", "sin_historial")
    no_elegibles = set(prod.loc[~prod["elegible"], "producto_id"])
    ids = [p for p, _ in r.items]
    assert len(ids) == 5 and not (set(ids) & no_elegibles)
    assert ids[0] == 100                       # el más popular y elegible
    pops = [modelo.pop[modelo.fila[i]] for i in ids]
    assert pops == sorted(pops, reverse=True)


def test_con_historial_personaliza_por_contenido(modelo, prod):
    joyas = sorted(ids_rubro(prod, "1"))
    r = modelo.recomendar([(joyas[0], "contacto")], [], n=5)
    assert (r.modo, r.motivo, r.n_perfil) == ("contenido", "historial", 1)
    ids = [p for p, _ in r.items]
    assert joyas[0] not in ids                 # no recomienda lo que ya conoce
    assert set(ids) <= ids_rubro(prod, "1")    # todo del mismo rubro


def test_no_recomienda_no_elegibles_ni_excluidos(modelo, prod):
    joyas = sorted(ids_rubro(prod, "1"))
    r0 = modelo.recomendar([(joyas[0], "vista")], n=50)
    excluido = r0.items[0][0]
    r1 = modelo.recomendar([(joyas[0], "vista")], n=50, excluir=[excluido])
    assert excluido in [p for p, _ in r0.items] and excluido not in [p for p, _ in r1.items]
    no_elegibles = set(prod.loc[~prod["elegible"], "producto_id"])
    assert not (set(p for p, _ in r0.items) & no_elegibles)


def test_sin_candidatos_si_ya_vio_todo_lo_elegible(modelo, prod):
    # perfil conocido y suficiente, pero no queda nada por recomendar: el motivo no debe ser "sin_historial"
    vistos = [(int(p), "vista") for p in prod.loc[prod["elegible"], "producto_id"]]
    r = modelo.recomendar(vistos, [], 10)
    assert (r.modo, r.motivo, r.items) == ("popularidad", "sin_candidatos", [])
    assert r.n_perfil == len(vistos)


def test_ids_desconocidos_y_tipos_invalidos_se_ignoran(modelo):
    r = modelo.recomendar([(999999, "vista"), (100, "tipo_raro")], n=3)
    assert r.modo == "popularidad" and r.ignorados == 2 and r.n_perfil == 0


def test_solo_busquedas_personaliza_aunque_no_haya_historial(modelo, prod):
    r = modelo.recomendar([], ["Joyería"], n=5)     # con tilde y mayúscula
    assert (r.modo, r.motivo, r.n_busquedas) == ("contenido", "solo_busquedas", 1)
    assert set(p for p, _ in r.items) <= ids_rubro(prod, "1")


def test_busqueda_sin_vocabulario_cae_a_popularidad(modelo):
    r = modelo.recomendar([], ["zzzqqq"], n=3)
    assert (r.modo, r.motivo, r.n_busquedas) == ("popularidad", "sin_historial", 0)


def test_busqueda_se_suma_al_historial(modelo, prod):
    joyas = sorted(ids_rubro(prod, "1"))
    ropa = sorted(ids_rubro(prod, "3"))
    sin = modelo.recomendar([(joyas[0], "vista")], [], n=3)
    con = modelo.recomendar([(joyas[0], "vista")], ["indumentaria", "remera"], n=3)
    assert set(p for p, _ in sin.items) <= ids_rubro(prod, "1")
    assert set(p for p, _ in con.items) & set(ropa)      # la búsqueda (peso 2 c/u) desplaza hacia ropa


def test_resultado_determinista(modelo, prod):
    args = ([(sorted(ids_rubro(prod, "2"))[0], "favorito")], ["chocolate"], 10)
    assert modelo.recomendar(*args).items == modelo.recomendar(*args).items


def test_maximo_por_producto_como_en_entrenamiento(modelo, prod):
    p = sorted(ids_rubro(prod, "2"))[0]
    a = modelo.recomendar([(p, "vista"), (p, "contacto")], n=5)
    b = modelo.recomendar([(p, "contacto")], n=5)
    assert a.items == b.items and a.n_perfil == 1


def test_equivalente_al_puntaje_offline(dict_modelo, modelo, prod):
    """El puntaje en línea coincide con score_contenido_busq que se usó para evaluar."""
    d, _, (T, C, P) = dict_modelo
    p1, p2 = sorted(ids_rubro(prod, "2"))[:2]
    inter = [(p1, "vista"), (p2, "favorito")]
    r = modelo.recomendar(inter, ["chocolate"], n=100)
    rows = np.array([modelo.fila[p1], modelo.fila[p2]])
    caso = {"prof_rows": rows, "prof_w": np.array([1.0, 3.0]), "q_rows": np.array([0])}
    b = d["busqueda"]
    Tq, Cq = bq.vectores_busqueda(["chocolate"], b["nombre_cat_a_id"], b["tfidf"], b["cats"])
    Q = bq.matriz_Q(Tq, Cq, (1, 0.5, 0.3), b["n_precio"])
    cand = np.array([modelo.fila[p] for p, _ in r.items])
    offline = bq.score_contenido_busq(caso, cand, None, X=modelo.X, Q=Q, wb=2.0)
    np.testing.assert_allclose([s for _, s in r.items], offline, atol=1e-5)


def test_recarga_en_caliente_con_rename_atomico(tmp_path):
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(construir_dict_modelo("v1")[0], ruta)
    st = ModeloStore(ruta, revisar_cada=0)
    assert st.get().version == "v1"
    tmp = tmp_path / "nuevo.tmp"
    joblib.dump(construir_dict_modelo("v2")[0], tmp)
    time.sleep(0.01)
    os.replace(tmp, ruta)
    assert st.get().version == "v2"


def test_recarga_fallida_mantiene_modelo_anterior(tmp_path):
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(construir_dict_modelo("v1")[0], ruta)
    st = ModeloStore(ruta, revisar_cada=0)
    time.sleep(0.01)
    ruta.write_bytes(b"corrupto")
    assert st.get().version == "v1"


def test_modelo_incompleto_se_rechaza():
    d = construir_dict_modelo()[0]
    d.pop("busqueda")
    with pytest.raises(ValueError):
        Modelo(d)


@pytest.mark.skipif(not Path("out/modelo.joblib").exists(), reason="falta out/modelo.joblib (correr modelo_eval.py)")
def test_contrato_con_el_modelo_real():
    d = joblib.load("out/modelo.joblib")
    assert all(k in d for k in CLAVES_MODELO)
    m = Modelo(d)
    r = m.recomendar([], ["gastronomia"], n=10)
    assert r.modo in ("contenido", "popularidad") and len(r.items) == 10
