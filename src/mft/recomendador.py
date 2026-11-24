"""
Núcleo del recomendador en línea: carga del modelo, reglas de arranque en frío y puntaje.

La API es SIN ESTADO de usuario: el backend manda las interacciones del usuario en cada pedido, así que la
VM del recomendador no guarda ni consulta datos personales. El modelo (modelo.joblib, generado cada noche por
modelo_eval.py) solo tiene vectores de productos, popularidad y la configuración.

Reglas (misma lógica que en la evaluación offline, ver notebooks/01_recomendador_bitacora.ipynb):
  1. El perfil es la suma ponderada de los vectores de los productos con los que el usuario interactuó
     (de cada producto se toma la señal de mayor peso) más los términos buscados (peso `peso_busqueda`).
  2. Hay personalización si el usuario tiene >= `umbral` productos conocidos por el modelo, o al menos una
     búsqueda utilizable. En ese caso se ordena por similitud coseno (modo "contenido").
  3. Si no hay nada con qué armar un perfil (arranque en frío), o el perfil no se parece a ningún candidato,
     se devuelven los productos más populares (modo "popularidad").
  4. Nunca se recomiendan productos no elegibles, ya conocidos por el usuario ni los que pida excluir el backend.
  5. Los empates se resuelven por popularidad y luego por id (resultado determinista).

Caso NO evaluado offline: usuarios sin productos pero con búsquedas (p. ej. anónimos con búsqueda de sesión).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

from . import busqueda as bq

log = logging.getLogger("mft.recomendador")

MAX_BUSQUEDAS = 5  # se usan las N búsquedas distintas más recientes
# Similitud mínima (coseno) para que un producto cuente como "parecido" a otro. Las filas de X están normalizadas, así que el
# producto punto es el coseno. El bloque de precio solo ya da ~0,07 entre productos sin nada en común: el umbral tiene que
# quedar por encima para no mostrar como parecidos a productos que solo caen en el mismo rango de precio.
MIN_SIMILITUD = float(os.environ.get("RECO_SIMILARES_MIN", "0.10"))   # ajustable sin tocar código
CLAVES_MODELO = ("version", "producto_ids", "elegible", "X", "popularidad", "config", "pesos_por_tipo", "busqueda")


@dataclass
class Resultado:
    modo: str                      # "contenido" | "popularidad"
    motivo: str                    # por qué se eligió ese modo
    items: list = field(default_factory=list)   # [(producto_id, score)]
    n_perfil: int = 0              # productos del usuario que el modelo conoce
    n_busquedas: int = 0           # búsquedas utilizables
    ignorados: int = 0             # producto_id que el modelo no conoce


class Modelo:
    def __init__(self, d: dict):
        faltan = [k for k in CLAVES_MODELO if k not in d]
        if faltan:
            raise ValueError(f"modelo.joblib incompleto, faltan: {faltan}")
        self.version = str(d["version"])
        self.ids = np.asarray(d["producto_ids"], dtype=np.int64)
        self.fila = {int(p): i for i, p in enumerate(self.ids)}
        self.elegible = np.asarray(d["elegible"], dtype=bool)
        self.X = d["X"].tocsr()
        self.pop = np.asarray(d["popularidad"], dtype=float)
        cfg = d["config"]
        self.pesos_bloques = tuple(cfg["pesos_bloques"])
        self.umbral = int(cfg["umbral"])
        self.wb = float(cfg["peso_busqueda"])
        self.pesos_tipo = dict(d["pesos_por_tipo"])
        self.busq = d["busqueda"]
        if not (len(self.ids) == self.X.shape[0] == len(self.elegible) == len(self.pop)):
            raise ValueError("modelo.joblib inconsistente: largos distintos")

    # ------------------------------------------------------------------ búsquedas
    def _vectores_busqueda(self, busquedas):
        vistos, terminos = set(), []
        for t in reversed(busquedas or []):          # las más recientes primero
            t = bq.norm_txt(t)
            if t and t not in vistos:
                vistos.add(t)
                terminos.append(t)
            if len(terminos) >= MAX_BUSQUEDAS:
                break
        if not terminos or self.wb <= 0:
            return None
        b = self.busq
        Tq, Cq = bq.vectores_busqueda(terminos, b["nombre_cat_a_id"], b["tfidf"], b["cats"])
        Q = bq.matriz_Q(Tq, Cq, self.pesos_bloques, b["n_precio"])
        util = np.diff(Q.indptr) > 0                 # descarta términos sin vocabulario ni categoría
        return Q[np.where(util)[0]] if util.any() else None

    # ------------------------------------------------------------------ recomendación
    def similares(self, producto_id, n=10):
        """Los productos más parecidos a uno dado: [(producto_id, score)] de mayor a menor similitud.

        Solo elegibles, nunca el mismo producto y solo con similitud >= MIN_SIMILITUD (puede devolver menos de n o
        ninguno). Desempate por popularidad y luego por id, para que el orden sea siempre el mismo.
        Devuelve None si el modelo no conoce el producto (p. ej. se creó después del último entrenamiento).
        """
        f = self.fila.get(int(producto_id))
        if f is None:
            return None
        score = (self.X @ self.X[f].T).toarray().ravel()
        ok = self.elegible & (score >= MIN_SIMILITUD)
        ok[f] = False
        cand = np.where(ok)[0]
        orden = np.lexsort((self.ids[cand], -self.pop[cand], -score[cand]))[:n]
        return [(int(self.ids[cand[i]]), round(float(score[cand[i]]), 6)) for i in orden]

    def recomendar(self, interacciones, busquedas=(), n=10, excluir=()) -> Resultado:
        """interacciones: iterable de (producto_id, tipo). Devuelve hasta n productos."""
        pesos = {}
        ignorados = 0
        for pid, tipo in interacciones:
            w = self.pesos_tipo.get(tipo)
            f = self.fila.get(int(pid))
            if w is None or f is None:
                ignorados += 1
                continue
            pesos[f] = max(pesos.get(f, 0.0), float(w))   # máximo por producto, como en el entrenamiento
        rows = np.array(sorted(pesos), dtype=int)
        w = np.array([pesos[r] for r in rows], dtype=float)
        Q = self._vectores_busqueda(busquedas)
        nq = 0 if Q is None else Q.shape[0]

        # candidatos: elegibles, no conocidos, no excluidos por el backend
        ok = self.elegible.copy()
        ok[rows] = False
        for pid in excluir:
            f = self.fila.get(int(pid))
            if f is not None:
                ok[f] = False
        cand = np.where(ok)[0]

        def top(score, modo, motivo):
            orden = np.lexsort((self.ids[cand], -self.pop[cand], -score))[:n]
            items = [(int(self.ids[cand[i]]), round(float(score[i]), 6)) for i in orden]
            return Resultado(modo, motivo, items, len(rows), nq, ignorados)

        personalizado = len(rows) >= self.umbral or nq > 0
        if personalizado and len(cand):
            u = sp.csr_matrix((1, self.X.shape[1]))
            if len(rows):
                u = u + sp.csr_matrix(w.reshape(1, -1)) @ self.X[rows]
            if nq:
                u = u + self.wb * (sp.csr_matrix(np.ones((1, nq))) @ Q)
            score = (self.X[cand] @ u.T).toarray().ravel()
            if score.max() > 0:
                motivo = "historial" if len(rows) >= self.umbral else "solo_busquedas"
                return top(score, "contenido", motivo)
            motivo = "perfil_sin_similitud"
        else:
            # personalizaría, pero no queda ningún candidato (todo lo elegible ya lo vio o está excluido)
            motivo = "sin_candidatos" if personalizado else "sin_historial"
        return top(self.pop[cand], "popularidad", motivo)


class ModeloStore:
    """Carga el modelo al iniciar y lo recarga solo cuando el archivo cambia (el job nocturno lo reemplaza
    con un rename atómico). Si la recarga falla se sigue sirviendo el modelo anterior."""

    def __init__(self, ruta, revisar_cada: float = 30.0):
        self.ruta = Path(ruta)
        self.revisar_cada = revisar_cada
        self._lock = threading.Lock()
        self._mtime = None
        self._ultimo_chequeo = 0.0
        self.cargado_en = None
        self.modelo = self._cargar()      # falla fuerte si no existe: mejor no arrancar sin modelo

    def _cargar(self) -> Modelo:
        mtime = self.ruta.stat().st_mtime_ns
        m = Modelo(joblib.load(self.ruta))
        self._mtime, self.cargado_en = mtime, time.time()
        log.info("modelo cargado: version=%s productos=%d", m.version, len(m.ids))
        return m

    def get(self) -> Modelo:
        ahora = time.monotonic()
        if ahora - self._ultimo_chequeo >= self.revisar_cada and self._lock.acquire(blocking=False):
            try:
                self._ultimo_chequeo = ahora
                if self.ruta.stat().st_mtime_ns != self._mtime:
                    self.modelo = self._cargar()    # asignación atómica de la referencia
            except Exception:
                log.exception("no se pudo recargar el modelo; se sigue con la versión anterior")
            finally:
                self._lock.release()
        return self.modelo


def ruta_modelo_por_defecto() -> str:
    return os.environ.get("RECO_MODEL_PATH", "models/current/modelo.joblib")
