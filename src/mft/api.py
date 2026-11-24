"""
API de recomendación (FastAPI).

  GET  /health       estado y versión del modelo (sin autenticación; para el health check del balanceador)
  POST /recomendar   recomendaciones para un usuario (header X-API-Key)
  POST /similares    productos parecidos a uno dado (header X-API-Key; no depende del usuario)

Variables de entorno:
  RECO_API_KEY      clave compartida con el backend (obligatoria)
  RECO_MODEL_PATH   ruta de modelo.joblib (por defecto models/current/modelo.joblib)

Ejecutar:  uv run uvicorn mft.api:app --host 127.0.0.1 --port 8000
"""
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .recomendador import ModeloStore, ruta_modelo_por_defecto

log = logging.getLogger("mft.api")


def _configurar_logs() -> None:
    """uvicorn solo configura sus propios loggers: sin esto, los log.info de mft.* (tiempos de cada pedido,
    recarga del modelo) no se ven ni en la consola ni en journald."""
    lg = logging.getLogger("mft")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False


_configurar_logs()


def _calentar(store) -> None:
    """La primera ejecución de cada camino de cálculo (importaciones perezosas de scipy/sklearn) puede tardar
    cientos de ms, sobre todo en máquinas lentas o cargadas. Se recorre una vez al arrancar para que el primer
    pedido real no pague ese costo. Nunca impide el arranque."""
    t0 = time.perf_counter()
    try:
        m = store.get()
        ids = [int(i) for i in m.ids[:5]]
        m.recomendar([(i, "vista") for i in ids], ["calentamiento"], 5, [])   # camino personalizado + búsqueda
        m.recomendar([], [], 5, [])                                              # camino de popularidad
        if len(m.ids):
            m.similares(int(m.ids[0]), 5)                                        # camino de similares
        log.info("calentamiento listo en %.0f ms", (time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        log.warning("el calentamiento falló (%s: %s); el primer pedido puede ser lento", type(e).__name__, e)


class Interaccion(BaseModel):
    producto_id: int
    tipo: Literal["vista", "compartido", "favorito", "calificacion", "contacto"]


class Pedido(BaseModel):
    user_id: str | None = Field(None, description="Solo para trazabilidad en el backend; la API no lo usa")
    interacciones_recientes: list[Interaccion] = Field(
        default_factory=list, max_length=500,
        description="Historial del usuario (hasta 500). Calificaciones: enviar solo las >= 4")
    busquedas: list[str] = Field(default_factory=list, max_length=50,
                                 description="Términos buscados, la más reciente al final")
    excluir: list[int] = Field(default_factory=list, max_length=2000,
                               description="Productos a no recomendar (p. ej. ya contactados)")
    n: int = Field(10, ge=1, le=50)


class Producto(BaseModel):
    producto_id: int
    score: float


class PedidoSimilares(BaseModel):
    producto_id: int
    n: int = Field(10, ge=1, le=50)


class RespuestaSimilares(BaseModel):
    modelo_version: str
    productos: list[Producto]


class Respuesta(BaseModel):
    modelo_version: str
    modo: Literal["contenido", "popularidad"]
    motivo: str
    n_perfil: int
    n_busquedas: int
    productos: list[Producto]


def crear_app(store=None, api_key: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "store", None) is None:
            app.state.store = ModeloStore(ruta_modelo_por_defecto())
        if not getattr(app.state, "api_key", None):
            app.state.api_key = os.environ.get("RECO_API_KEY", "")
        if not app.state.api_key:
            raise RuntimeError("Falta RECO_API_KEY")
        _calentar(app.state.store)
        yield

    app = FastAPI(title="MercadoFT recomendador", version="1.0", lifespan=lifespan)
    app.state.store = store
    app.state.api_key = api_key

    def autenticar(x_api_key: str | None = Header(None)):
        esperado = app.state.api_key or ""
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), esperado.encode()):
            raise HTTPException(status_code=401, detail="API key inválida")

    @app.get("/health")
    def health():
        st = app.state.store
        if st is None:
            raise HTTPException(status_code=503, detail="modelo no cargado")
        m = st.get()
        try:
            edad_dias = (date.today() - date.fromisoformat(m.version)).days   # para alertar si el modelo envejece
        except ValueError:
            edad_dias = None
        return {"estado": "ok", "modelo_version": m.version, "edad_dias": edad_dias, "productos": int(len(m.ids)),
                "elegibles": int(m.elegible.sum()), "cargado_en": st.cargado_en}

    @app.post("/recomendar", response_model=Respuesta, dependencies=[Depends(autenticar)])
    def recomendar(p: Pedido):
        t0 = time.perf_counter()
        m = app.state.store.get()
        r = m.recomendar([(i.producto_id, i.tipo) for i in p.interacciones_recientes],
                         p.busquedas, p.n, p.excluir)
        log.info("recomendar modo=%s motivo=%s perfil=%d busquedas=%d devueltos=%d ms=%.1f",
                 r.modo, r.motivo, r.n_perfil, r.n_busquedas, len(r.items), (time.perf_counter() - t0) * 1000)
        return Respuesta(modelo_version=m.version, modo=r.modo, motivo=r.motivo, n_perfil=r.n_perfil,
                         n_busquedas=r.n_busquedas,
                         productos=[Producto(producto_id=a, score=b) for a, b in r.items])

    @app.post("/similares", response_model=RespuestaSimilares, dependencies=[Depends(autenticar)])
    def similares(p: PedidoSimilares):
        """Productos parecidos a uno dado. No depende del usuario, así que el backend puede cachear la respuesta."""
        t0 = time.perf_counter()
        m = app.state.store.get()
        items = m.similares(p.producto_id, p.n)
        if items is None:
            raise HTTPException(status_code=404, detail="producto desconocido para el modelo")
        log.info("similares producto=%d devueltos=%d ms=%.1f", p.producto_id, len(items),
                 (time.perf_counter() - t0) * 1000)
        return RespuestaSimilares(modelo_version=m.version,
                                  productos=[Producto(producto_id=a, score=b) for a, b in items])

    return app


app = crear_app()
