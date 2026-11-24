import numpy as np
import pandas as pd
import pytest

import modelo_eval as me
from mft import busqueda as bq
from mft.recomendador import Modelo

PESOS_TIPO = {"vista": 1.0, "compartido": 2.0, "favorito": 3.0, "calificacion": 3.0, "contacto": 5.0}

RUBROS = {
    "1": ("Joyería", ["anillo de plata", "collar artesanal", "aros de oro", "pulsera de plata", "dije de oro",
                      "anillo con piedra", "collar de perlas", "aros de plata"], "joya plata oro artesanal piedra"),
    "2": ("Gastronomía", ["torta de chocolate", "alfajores caseros", "mermelada de frutilla", "pan de masa madre",
                          "brownie con nueces", "dulce de leche", "budín de limón", "galletitas de avena"],
          "comida dulce casera receta ingredientes"),
    "3": ("Indumentaria", ["remera de algodón", "buzo con capucha", "pantalón de jean", "campera de abrigo",
                           "vestido de verano", "camisa de lino", "short deportivo", "gorra bordada"],
          "ropa prenda talle tela algodón"),
}


def construir_dict_modelo(version="test"):
    filas, pid = [], 100
    for cid, (nombre, items, extra) in RUBROS.items():
        for i, it in enumerate(items):
            filas.append({"producto_id": pid, "nombre": it, "descripcion": f"{it} {extra} hecho a mano {nombre.lower()}",
                          "categorias": cid, "categorias_nombres": nombre, "precio": 1000 + 700 * i + 50 * pid,
                          "cid": cid})
            pid += 1
    prod = pd.DataFrame(filas)
    prod["elegible"] = True
    prod.loc[[3, 11], "elegible"] = False          # dos productos no elegibles
    T, C, P, tfidf, cats = me.bloques_features(prod)
    w = (1, 0.5, 0.3)
    X = me.armar_X(T, C, P, w)
    pop = np.linspace(0, 10, len(prod))[::-1].copy()   # el producto 100 es el más popular
    d = {"version": version, "producto_ids": prod["producto_id"].to_numpy(), "elegible": prod["elegible"].to_numpy(),
         "X": X, "popularidad": pop,
         "config": {"pesos_bloques": w, "esquema": "pesos", "umbral": 1, "peso_busqueda": 2},
         "pesos_por_tipo": PESOS_TIPO,
         "busqueda": {"tfidf": tfidf, "cats": cats, "nombre_cat_a_id": bq.mapa_categorias(prod), "n_precio": int(P.shape[1])}}
    return d, prod, (T, C, P)


@pytest.fixture(scope="session")
def dict_modelo():
    return construir_dict_modelo()


@pytest.fixture(scope="session")
def modelo(dict_modelo):
    return Modelo(dict_modelo[0])


@pytest.fixture(scope="session")
def prod(dict_modelo):
    return dict_modelo[1]
