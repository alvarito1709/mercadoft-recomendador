# Medición de "Productos similares"

Cada vista que llega desde el carrusel de similares se guarda en `producto_visto` con
`origen = 'similares'` y `producto_origen_id` = el producto desde el que se tocó. Las vistas que no
vienen de ahí tienen `origen` nulo, igual que todas las anteriores a esta medición.

## Qué se mide y qué no

- Se mide: cuántas vistas llegan desde similares y entre qué pares de productos.
- No se mide el CTR: no se registra cuántas veces se mostró la lista (el feed sí tiene
  `recomendacion_servida`; los similares no).
- Solo cuentan usuarios con sesión (`marcar-vista` necesita usuario) y no cuentan las vistas del
  propio emprendedor.
- Si el recomendador está apagado, la lista sale solo por categoría y los clics igual se registran
  con `origen = 'similares'`. Para separar modelo de categoría habría que guardar también la
  fuente (`X-Reco-Fuente`); no está hecho.

## Consultas (MySQL)

Fecha de inicio de la medición: la primera vista con `origen` no nulo.

```sql
SELECT MIN(fecha) AS desde FROM producto_visto WHERE origen IS NOT NULL;
```

Porcentaje de vistas que vienen de similares (desde que empezó la medición):

```sql
SELECT SUM(origen = 'similares')                                AS desde_similares,
       COUNT(*)                                                 AS total_vistas,
       ROUND(100 * SUM(origen = 'similares') / COUNT(*), 2)     AS porcentaje
FROM producto_visto
WHERE fecha >= (SELECT MIN(fecha) FROM producto_visto WHERE origen IS NOT NULL);
```

Pares más tocados (producto de origen → producto de destino):

```sql
SELECT producto_origen_id,
       COALESCE(producto_id, producto_original_id) AS producto_destino_id,
       COUNT(*)                                    AS clics
FROM producto_visto
WHERE origen = 'similares'
GROUP BY producto_origen_id, COALESCE(producto_id, producto_original_id)
ORDER BY clics DESC
LIMIT 50;
```

De las vistas desde similares, cuántas terminan en consulta por WhatsApp (mismo usuario y mismo
producto, dentro de las 24 horas siguientes):

```sql
SELECT COUNT(*)                                          AS vistas_desde_similares,
       SUM(EXISTS (SELECT 1 FROM consulta c
                   WHERE c.usuario_id  = pv.usuario_id
                     AND c.producto_id = pv.producto_id
                     AND c.fecha >= pv.fecha
                     AND c.fecha <  pv.fecha + INTERVAL 1 DAY)) AS con_consulta
FROM producto_visto pv
WHERE pv.origen = 'similares';
```

## Para la sección de limitaciones de la tesis

Los productos que el modelo muestra reciben más vistas, y esas vistas después entran al
entrenamiento: hay un sesgo de retroalimentación. Las vistas desde similares quedan marcadas, por lo
que se pueden separar del resto si hace falta.
