# Contrato del snapshot (backend → bucket → VM del recomendador)

Cada noche el backend (Spring Boot) deja en un bucket **privado** una carpeta con una foto de las tablas que usa el
recomendador. La VM del recomendador la baja, la valida y entrena. La VM nunca toca la base.

La definición ejecutable de este contrato está en `src/mft/snapshot.py` (`CONTRATO`, `PROHIBIDAS`, `validar`).
Si este documento y el código difieren, manda el código.

## Estructura

```
gs://<bucket-snapshots>/snapshots/2026-11-14/
    producto.csv  producto_categoria.csv  categoria.csv  emprendedor.csv  producto_borrado.csv  usuario.csv
    producto_visto.csv  favorito.csv  consulta.csv  calificacion.csv  compartido.csv  busqueda.csv
    manifest.json        <- se escribe AL FINAL
```

- Carpeta `AAAA-MM-DD` (fecha del export). Si se reexporta el mismo día se sobrescribe, y el manifest se escribe de nuevo al final.
- **El manifest es la señal de "snapshot completo".** Sin manifest, el job lo ignora (puede estar a medio escribir).
- Con gcsfuse un archivo aparece en el bucket al **cerrarse**: cerrar todos los CSV antes de escribir el manifest.

## Formato de los CSV

- UTF-8, separador coma, comillas dobles cuando haga falta (las descripciones tienen comas y saltos de línea), fila de encabezado en minúsculas.
- Todas las tablas son obligatorias, aunque estén vacías (solo el encabezado).
- Booleanos: `0` / `1`. Nulos: campo vacío.
- Fechas: `AAAA-MM-DD HH:MM:SS` (se acepta fracción de segundo) **en la misma zona horaria para todas las tablas** (la de la base).
- Ids numéricos tal como están en la base. Se pueden exportar columnas extra: se descartan, salvo las prohibidas (abajo).

## Tablas y columnas

| Tabla | Columnas |
|---|---|
| `producto` | id, nombre, descripcion, precio, desactivada, estado_revision, fecha_carga, emprendedor_id |
| `producto_categoria` | producto_id, categoria_id |
| `categoria` | id, nombre |
| `emprendedor` | id, usuario_id, estado_aprobacion |
| `producto_borrado` | producto_original_id, nombre, descripcion, precio, emprendedor_id, fecha_carga |
| `usuario` | id, eliminado, email_verificado, rol_id, fecha_de_registro |
| `producto_visto` | fecha, producto_original_id, producto_id, usuario_id |
| `favorito` | fecha_creado, producto_id, usuario_id |
| `consulta` | fecha, producto_original_id, producto_id, usuario_id |
| `calificacion` | calificacion, fecha, producto_original_id, producto_id, usuario_id |
| `compartido` | fecha, tipo, producto_original_id, producto_id, usuario_id |
| `busqueda` | fecha_busqueda, termino_busqueda, usuario_id |

Notas:
- Se exportan **todas las filas** (no solo las del día): el modelo se entrena con la historia completa.
- `consulta` es el clic en "contactar al vendedor". `compartido.tipo` distingue PRODUCTO de EMPRENDEDOR.
- `busqueda.usuario_id` puede ser nulo (búsquedas anónimas). Exportarlas igual; el entrenamiento las descarta.
- `producto_visto`, `favorito`, `consulta`, `calificacion` y `compartido` pueden tener `producto_id` nulo con `producto_original_id` informado (producto borrado).

## Columnas PROHIBIDAS (el snapshot se rechaza si aparecen)

`email`, `password`, `documento`, `numero_celular`, `nombres`, `apellidos`, `username`, `comentario`,
`motivo_rechazo`, `facebook`, `instagram`, `imagen_perfil`, `nombre_imagen_perfil`, `anonimo_id`.

Las exportaciones manuales que se usaron durante el desarrollo **sí** traían varias de estas columnas
(`usuario`: email, password, nombres…; `emprendedor`: documento, numero_celular…). El export automático no debe incluirlas:
seleccionar solo las columnas del contrato.

## manifest.json

```json
{
  "formato": 1,
  "generado": "2026-11-14T03:35:12-03:00",
  "tablas": {
    "producto":       {"archivo": "producto.csv",       "filas": 856,  "sha256": "<hex, opcional>"},
    "producto_visto": {"archivo": "producto_visto.csv", "filas": 3763, "sha256": "<hex, opcional>"}
  }
}
```

- `formato`: entero, hoy `1`.
- `generado`: fecha-hora ISO-8601 **con zona horaria**.
- `tablas`: una entrada por cada tabla del contrato. `filas` es la cantidad de filas de datos (sin encabezado); el job
  rechaza el snapshot si no coincide. `sha256` es opcional, pero recomendado: detecta archivos truncados.

## Qué hace el job con un snapshot

1. Toma el más reciente que tenga manifest.
2. Lo valida (manifest, tablas, columnas, filas, hashes, columnas prohibidas). Si falla, **no toca el modelo publicado** y sale con error.
3. Lo reduce a las columnas del contrato, corre el pipeline y entrena.
4. Prueba el modelo (humo + caída de eventos/productos elegibles respecto de la versión publicada) y recién ahí lo publica con un rename atómico.
5. Borra el crudo descargado; conserva las últimas 7 versiones del modelo para poder volver atrás.

## Recomendaciones para el bucket

- Bucket **dedicado y privado** (acceso uniforme, prevención de acceso público forzada). No usar el bucket público de imágenes.
- Backend: puede crear objetos; VM del recomendador: solo lectura.
- Regla de ciclo de vida para borrar snapshots de más de 30 días.
