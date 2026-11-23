# Despliegue en la VM (e2-medium)

Una sola VM sirve las dos instancias del recomendador, **dev** y **prod**, con unidades systemd de plantilla
(`mft-api@dev`, `mft-api@prod`, `mft-nocturno@dev`, ...). Cada instancia tiene su archivo `/etc/mft/<instancia>.env`,
su puerto, su carpeta de modelos y su carpeta del bucket.

| | dev | prod |
|---|---|---|
| Puerto de la API | 8001 | 8000 |
| Workers | 1 | 2 |
| Carpeta de trabajo y modelos (`RECO_BASE`) | `/var/lib/mft/dev` | `/var/lib/mft/prod` |
| Origen de snapshots (`RECO_ORIGEN`) | `gs://training_snapshots/DEV` | `gs://training_snapshots/PROD` |

Esta guía arma **dev**. Para prod se repite todo con `prod` en lugar de `dev` (otra API key, puerto 8000, 2 workers).

```
backend (Tomcat, VM del backend) --HTTP, IP interna--> API del recomendador (esta VM, 8001/8000)
backend --gcsfuse--> bucket training_snapshots/DEV --gcloud storage cp--> job nocturno (esta VM) --> modelo --> API
```

La VM del backend y esta VM están en el mismo proyecto y se hablan por IP interna: no hace falta Nginx ni HTTPS
mientras la API no se exponga a internet. La protección es la **API key** más el **firewall**.

---

## 1. Red

Las dos VMs tienen que estar en la misma red de VPC (normalmente `default`).

```bash
gcloud compute instances list --format="table(name,zone.basename(),networkInterfaces[0].network.basename(),networkInterfaces[0].networkIP,tags.items.list())"
```

- Si las dos muestran la misma red (por ejemplo `default`), **no hay que configurar nada más**: ya se ven por la IP interna
  (`10.x.x.x`). La red `default` trae la regla `default-allow-internal`, que permite el tráfico entre VMs.
  Se confirma con `gcloud compute firewall-rules list --filter="network:default" --format="table(name,sourceRanges.list(),allowed[].map().firewall_rule().list())"`.
- Si las redes son distintas, o esa regla no existe, crear una regla que deje entrar **solo** al backend:

```bash
gcloud compute firewall-rules create mft-reco-desde-backend --network=default --direction=INGRESS \
  --allow=tcp:8000,tcp:8001 --source-ranges=<IP_INTERNA_VM_BACKEND>/32 --target-tags=mft-reco
gcloud compute instances add-tags <NOMBRE_VM_RECOMENDADOR> --zone=<ZONA> --tags=mft-reco
```

No crear nunca una regla que abra 8000/8001 a `0.0.0.0/0`.

## 2. Permiso de lectura del bucket

La VM baja los snapshots con la cuenta de servicio de la propia VM.

```bash
gcloud compute instances describe <NOMBRE_VM_RECOMENDADOR> --zone=<ZONA> --format="value(serviceAccounts[0].email,serviceAccounts[0].scopes)"
```

- Los scopes tienen que incluir `devstorage.read_only` (o `devstorage.full_control` / `cloud-platform`).
  Cambiarlos requiere apagar la VM.
- Dar lectura sobre el bucket a esa cuenta:

```bash
gcloud storage buckets add-iam-policy-binding gs://training_snapshots \
  --member="serviceAccount:<EMAIL_DE_LA_CUENTA>" --role="roles/storage.objectViewer"
```

## 3. Preparar la VM (por SSH)

```bash
sudo apt-get update && sudo apt-get install -y git curl
sudo useradd --system --create-home --shell /usr/sbin/nologin mft
sudo mkdir -p /opt/mercadoft-recomendador /var/lib/mft /etc/mft
sudo chown mft:mft /opt/mercadoft-recomendador /var/lib/mft

# uv instalado para todos los usuarios (maneja Python 3.12 y las dependencias)
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
```

Código (el repo es privado: usar un token de GitHub de solo lectura, permiso *Contents: read*, y sacarlo de la URL al terminar):

```bash
sudo -H -u mft git clone --branch <RAMA> https://<USUARIO>:<TOKEN>@github.com/alvarito1709/mercadoft-recomendador.git /opt/mercadoft-recomendador
sudo -H -u mft git -C /opt/mercadoft-recomendador remote set-url origin https://github.com/alvarito1709/mercadoft-recomendador.git
cd /opt/mercadoft-recomendador
sudo -H -u mft uv sync --no-dev --frozen
```

## 4. Archivo de la instancia

```bash
sudo cp jobs/systemd/instancia.env.ejemplo /etc/mft/dev.env
sudo chown root:mft /etc/mft/dev.env && sudo chmod 640 /etc/mft/dev.env
openssl rand -hex 32     # esta es la API key: va en RECO_API_KEY y también en el backend
hostname -I              # IP interna de la VM: va en RECO_HOST
sudo nano /etc/mft/dev.env
sudo -u mft mkdir -p /var/lib/mft/dev
```

Completar `RECO_API_KEY` y `RECO_HOST`. El resto de los valores de `dev` ya vienen cargados.

## 5. Primer snapshot en el bucket

Mientras el backend de DEV no exporte solo, se puede subir uno generado en local (desde tu PC, con la carpeta que dejó
el exportador, por ejemplo `AAAA-MM-DD/` con sus 12 CSV y el `manifest.json`):

```bash
gcloud storage cp --recursive <CARPETA_LOCAL>/<AAAA-MM-DD> gs://training_snapshots/DEV/
gcloud storage ls gs://training_snapshots/DEV/<AAAA-MM-DD>/manifest.json
```

Que el manifest quede dentro de la carpeta con fecha (no suelto en `DEV/`). Un snapshot viejo solo genera una advertencia
en la primera corrida; no la bloquea.

## 6. Instalar las unidades y entrenar el primer modelo

```bash
cd /opt/mercadoft-recomendador
sudo install -m 644 jobs/systemd/mft-api@.service jobs/systemd/mft-nocturno@.service jobs/systemd/mft-nocturno@.timer /etc/systemd/system/
sudo systemctl daemon-reload

sudo systemctl start mft-nocturno@dev.service        # oneshot: espera a que termine
journalctl -u mft-nocturno@dev -n 60 --no-pager
ls /var/lib/mft/dev/models/current/                    # tiene que aparecer modelo.joblib
```

Salida del job: 0 ok o sin novedades · 1 error · 2 snapshot desactualizado · 3 ya hay otra corrida.

## 7. Levantar la API y probarla

```bash
sudo systemctl enable --now mft-api@dev
journalctl -u mft-api@dev -n 30 --no-pager            # busca "calentamiento listo"

IP=$(hostname -I | awk '{print $1}')
curl -s http://$IP:8001/health
curl -s -X POST http://$IP:8001/recomendar -H "X-API-Key: <API_KEY>" -H "Content-Type: application/json" \
  -d '{"interacciones_recientes":[{"producto_id":1,"tipo":"vista"}],"n":5}'
```

`/health` informa `modelo_version` y `edad_dias`. Después probar **desde la VM del backend** (`curl http://<IP_INTERNA>:8001/health`):
si responde, la red está bien. Si no responde, volver al paso 1.

## 8. Timer del job nocturno

```bash
sudo systemctl enable --now mft-nocturno@dev.timer
systemctl list-timers | grep mft
```

Corre todos los días a las 04:30 (hora de Buenos Aires). El backend exporta a las 03:30. La API recarga el modelo sola
cuando el archivo cambia (revisa cada 30 s), no hace falta reiniciarla.

## 9. Backend de DEV

Variables de entorno de Tomcat (junto a las que ya tiene):

```
RECO_API_ENABLED=true
RECO_API_URL=http://<IP_INTERNA_VM_RECOMENDADOR>:8001
RECO_API_KEY=<la misma clave de /etc/mft/dev.env>
RECO_EXPORT_ENABLED=true
RECO_EXPORT_DIR=<carpeta donde está montado gs://training_snapshots/DEV, con escritura>
```

Y encender el recomendador desde la web administrativa (o con
`UPDATE configuracion_recomendador SET habilitado = 1, porcentaje_usuarios = 100 WHERE id = 1`).
Chequeo: pedir `listarAleatorio` con un usuario con sesión; la respuesta trae el header `X-Reco-Fuente`.

## Operación

| Qué | Cómo |
|---|---|
| Ver logs de la API | `journalctl -u mft-api@dev -f` |
| Ver el último entrenamiento | `journalctl -u mft-nocturno@dev -n 100 --no-pager` |
| Entrenar a mano | `sudo systemctl start mft-nocturno@dev.service` |
| Volver al modelo anterior | `sudo -u mft bash -c 'set -a; . /etc/mft/dev.env; set +a; /opt/mercadoft-recomendador/jobs/nocturno.sh --rollback'` |
| Actualizar el código | `cd /opt/mercadoft-recomendador && sudo -H -u mft git pull && sudo -H -u mft uv sync --no-dev --frozen && sudo systemctl restart mft-api@dev` |
| Rotar la API key | cambiarla en `/etc/mft/dev.env` y en Tomcat, y `sudo systemctl restart mft-api@dev` |

## Problemas frecuentes

- **401 en el backend:** la `RECO_API_KEY` de Tomcat no es la del archivo de la instancia.
- **`/health` devuelve 503 "modelo no cargado":** todavía no hay modelo; correr el paso 6 y revisar los logs del job.
- **El backend no llega a la API:** revisar el paso 1; confirmar que `RECO_HOST` es la IP interna correcta y que la API escucha ahí (`ss -ltnp | grep 8001`).
- **El job falla con permisos sobre el bucket:** revisar el paso 2 (cuenta de servicio y scopes).
- **El job termina con código 2:** el backend dejó de exportar; mirar los logs del backend (`[reco-export]`).
