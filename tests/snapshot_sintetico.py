"""Genera un snapshot sintético que cumple el contrato (para probar el job sin datos reales)."""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from mft import snapshot as sn

RUBROS = {1: "Joyería", 2: "Gastronomía", 3: "Indumentaria"}
PALABRAS = {1: "anillo collar plata oro piedra", 2: "torta chocolate dulce casero harina", 3: "remera algodón talle prenda tela"}


def _w(carpeta, tabla, filas):
    with open(Path(carpeta) / f"{tabla}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(sn.CONTRATO[tabla])
        w.writerows(filas)


def crear_snapshot(carpeta, generado="2026-11-01T03:30:00-03:00", semilla=1, factor_eventos=1.0):
    rnd = random.Random(semilla)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    _w(carpeta, "categoria", [[c, n] for c, n in RUBROS.items()])
    productos, pc = [], []
    for pid in range(1, 31):
        cat = (pid - 1) % 3 + 1
        productos.append([pid, f"{PALABRAS[cat].split()[pid % 5]} {pid}", f"{PALABRAS[cat]} hecho a mano número {pid}",
                          1000 + 100 * pid, 0, "APROBADO", "2026-06-01 10:00:00", 1 + pid % 5])
        pc.append([pid, cat])
    productos[4][4] = 1                                   # un producto desactivado
    _w(carpeta, "producto", productos)
    _w(carpeta, "producto_categoria", pc)
    _w(carpeta, "emprendedor", [[e, 100 + e, "APROBADO"] for e in range(1, 6)])
    _w(carpeta, "producto_borrado", [])
    usuarios = [[u, 0, 1, 3, "2026-06-10"] for u in range(1, 61)] + [[100 + e, 0, 1, 3, "2026-06-10"] for e in range(1, 6)]
    _w(carpeta, "usuario", usuarios)
    base = datetime(2026, 7, 1)
    vistas, favs, cons, cal, busq = [], [], [], [], []
    for u in range(1, 61):
        gusto = u % 3 + 1
        propios = [p[0] for p in productos if (p[0] - 1) % 3 + 1 == gusto]
        n = max(1, int(rnd.randint(3, 12) * factor_eventos))
        for _ in range(n):
            pid = rnd.choice(propios)
            t = (base + timedelta(days=rnd.randint(0, 120), hours=rnd.randint(0, 23))).strftime("%Y-%m-%d %H:%M:%S")
            vistas.append([t, "", pid, u])
            if rnd.random() < 0.2:
                favs.append([t, pid, u])
            if rnd.random() < 0.15:
                cons.append([t, "", pid, u])
        if rnd.random() < 0.3:
            busq.append([(base + timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S"), RUBROS[gusto].lower(), u])
    cal.append([5, "2026-09-01 10:00:00", "", 3, 1])
    _w(carpeta, "producto_visto", vistas)
    _w(carpeta, "favorito", favs)
    _w(carpeta, "consulta", cons)
    _w(carpeta, "calificacion", cal)
    _w(carpeta, "compartido", [])
    _w(carpeta, "busqueda", busq)
    sn.escribir_manifest(carpeta, generado)
    return carpeta
