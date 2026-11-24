# MercadoFT – Sistema de recomendación (basado en contenido)

TFI de la tecnicatura en ciencia de datos e IA. La documentación del proceso y de las decisiones está en
[`notebooks/01_recomendador_bitacora.ipynb`](notebooks/01_recomendador_bitacora.ipynb).

## Setup (reproducible con uv)

Requiere [uv](https://docs.astral.sh/uv/). Python 3.12 se instala solo.

```powershell
# Windows (PowerShell): instalar uv si no lo tenés
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync            # crea .venv e instala todo desde uv.lock (incluye el paquete src/mft)
```

Alternativa sin uv (solo runtime): `python -m venv .venv`, activarlo, `pip install -r requirements.txt` y `pip install -e . --no-deps`.

## Pipeline

```
uv run python eda_recomendador.py --data-dir data --out out      # 1. EDA y armado de eventos/productos
uv run python modelo_eval.py --data-dir data --in out --out out  # 2. modelo + evaluación + out/modelo.joblib
uv run pytest                                                    # tests
```

`data/` (CSV de producción), `out/` (derivados, `out/.salt`, modelo) y `*.joblib` **no se versionan**.
La configuración final del modelo está en `CONFIG_FINAL` (`modelo_eval.py`); con `--auto` se elige por máximo NDCG@10.

## API

Código en `src/mft/` (`recomendador.py` = reglas y puntaje, `api.py` = FastAPI).

```
RECO_API_KEY=<clave> RECO_MODEL_PATH=out/modelo.joblib uv run uvicorn mft.api:app --host 127.0.0.1 --port 8000
```

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado y versión del modelo (sin API key) |
| `POST /recomendar` | Recomendaciones (header `X-API-Key`) |
| `POST /similares` | Productos parecidos a uno dado (header `X-API-Key`) |

```json
POST /recomendar
{
  "user_id": "opcional, solo trazabilidad",
  "interacciones_recientes": [{"producto_id": 123, "tipo": "vista|compartido|favorito|calificacion|contacto"}],
  "busquedas": ["chocolate", "joyería"],
  "excluir": [45, 67],
  "n": 10
}
→ {"modelo_version": "...", "modo": "contenido|popularidad", "motivo": "...", "n_perfil": 7, "n_busquedas": 2,
   "productos": [{"producto_id": 88, "score": 0.41}]}
```

- La API **no guarda estado de usuario**: el backend manda el historial (hasta 500 interacciones; calificaciones solo ≥ 4) en cada pedido.
- **Arranque en frío:** sin productos conocidos ni búsquedas utilizables → `popularidad` (`motivo: sin_historial`). Con al menos un producto
  conocido o una búsqueda utilizable → `contenido`. Si el perfil no se parece a ningún candidato → `popularidad` (`perfil_sin_similitud`).
- Nunca devuelve productos no elegibles, ya conocidos por el usuario ni los de `excluir`.
- **`POST /similares`** `{"producto_id": 123, "n": 10}` → `{"modelo_version": "...", "productos": [{"producto_id": 88, "score": 0.41}]}`.
  No depende del usuario (el backend puede cachear la respuesta). Devuelve solo productos elegibles, nunca el mismo, con similitud
  (coseno) mayor o igual a `RECO_SIMILARES_MIN` (por defecto 0.10; puede devolver menos de `n` o ninguno). `404` si el modelo no
  conoce el producto (p. ej. se creó después del último entrenamiento). Costo: ~1 ms con 5.000 productos.
- El modelo se recarga solo cuando `modelo.joblib` cambia en disco (cada ≤ 30 s); el job nocturno debe reemplazarlo con
  un rename atómico. Si la recarga falla, sigue sirviendo la versión anterior.

## Job nocturno (snapshot → modelo)

El backend deja cada noche un snapshot en un bucket privado (contrato en [`docs/contrato_snapshot.md`](docs/contrato_snapshot.md)).
La VM lo baja con `gcloud storage cp`, lo valida, entrena y publica el modelo con un rename atómico. Si algo falla, **el modelo
que sirve la API no se toca**.

```
# prueba local con una carpeta que contiene AAAA-MM-DD/ (sin bucket)
uv run python -m mft.job_nocturno --origen ./snapshots --base ./var

# en la VM: guía paso a paso en docs/despliegue_vm.md (unidades systemd de plantilla en jobs/systemd/)
# a mano, con RECO_ORIGEN y RECO_BASE
RECO_ORIGEN=gs://<bucket>/snapshots RECO_BASE=/var/lib/mft jobs/nocturno.sh
jobs/nocturno.sh --modo evaluar          # corrida completa con evaluación (guarda informes/)
jobs/nocturno.sh --rollback              # volver a la versión anterior (o --rollback AAAA-MM-DD)
```

Salida: 0 ok o sin novedades · 1 error · 2 snapshot desactualizado (>48 h) · 3 ya hay otra corrida.
La API debe apuntar a `RECO_MODEL_PATH=<base>/models/current/modelo.joblib`; `/health` informa `edad_dias` del modelo.

## Dependencias

- Agregar: `uv add <paquete>` (o `uv add --dev <paquete>`).
- Regenerar requirements.txt: `uv export --no-hashes --no-emit-project --no-annotate --no-dev -o requirements.txt`
