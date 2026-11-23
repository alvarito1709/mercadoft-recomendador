import pytest
from fastapi.testclient import TestClient

from mft.api import crear_app
from mft.recomendador import Modelo
from tests.conftest import construir_dict_modelo

KEY = "clave-de-prueba"


class FakeStore:
    cargado_en = 0.0

    def __init__(self, modelo):
        self.modelo = modelo

    def get(self):
        return self.modelo


@pytest.fixture()
def cliente(modelo):
    return TestClient(crear_app(store=FakeStore(modelo), api_key=KEY))


H = {"X-API-Key": KEY}


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["modelo_version"] == "test" and r.json()["elegibles"] == 22


def test_requiere_api_key(cliente):
    assert cliente.post("/recomendar", json={}).status_code == 401
    assert cliente.post("/recomendar", json={}, headers={"X-API-Key": "otra"}).status_code == 401


def test_cuerpo_vacio_devuelve_populares(cliente):
    r = cliente.post("/recomendar", json={}, headers=H)
    assert r.status_code == 200
    j = r.json()
    assert j["modo"] == "popularidad" and len(j["productos"]) == 10


def test_con_historial_y_exclusiones(cliente):
    body = {"interacciones_recientes": [{"producto_id": 100, "tipo": "contacto"}], "n": 4, "excluir": [101]}
    j = cliente.post("/recomendar", json=body, headers=H).json()
    ids = [p["producto_id"] for p in j["productos"]]
    assert j["modo"] == "contenido" and len(ids) == 4 and 100 not in ids and 101 not in ids


def test_validaciones(cliente):
    assert cliente.post("/recomendar", json={"n": 0}, headers=H).status_code == 422
    assert cliente.post("/recomendar", json={"n": 51}, headers=H).status_code == 422
    malo = {"interacciones_recientes": [{"producto_id": 1, "tipo": "like"}]}
    assert cliente.post("/recomendar", json=malo, headers=H).status_code == 422


def test_arranque_exige_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("RECO_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        with TestClient(crear_app(store=FakeStore(Modelo(construir_dict_modelo()[0])))):
            pass
