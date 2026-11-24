"""
Señal de búsqueda para el modelo de contenido (módulo compartido por modelo_eval.py y eval_busqueda.py).

Cada término buscado por el usuario (antes del producto objetivo) se convierte en un vector del mismo espacio
que los productos: TF-IDF del término + categoría si el término es exactamente el nombre de una categoría.
Se suma al perfil con un peso w_b (w_b = 0 -> modelo sin búsqueda).

Solo entran búsquedas con usuario_id (las anónimas no se pueden atar a un perfil), de usuarios presentes en
eventos.csv, y cada (usuario, término) cuenta una sola vez (la primera aparición).
"""
import hashlib
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def norm_txt(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return " ".join(s.split())


def hash_id(x, salt):
    return hashlib.sha256((salt + str(int(x))).encode()).hexdigest()[:12]


def cargar_busquedas(data_dir: Path, out_dir: Path, usuarios_validos):
    """Devuelve (DataFrame[usuario, termino, fecha], info) o (None, motivo) si no se puede usar."""
    f = Path(data_dir) / "busqueda.csv"
    if not f.exists():
        return None, f"no existe {f}"
    salt = os.environ.get("RECO_SALT")
    if not salt:
        fs = Path(out_dir) / ".salt"
        if not fs.exists():
            return None, "falta el salt (out/.salt o variable RECO_SALT)"
        salt = fs.read_text().strip()
    b = pd.read_csv(f)
    total = len(b)
    b = b.dropna(subset=["usuario_id"]).copy()
    anon = total - len(b)
    b["usuario"] = b["usuario_id"].map(lambda x: hash_id(x, salt))
    b = b[b["usuario"].isin(usuarios_validos)]
    b["fecha"] = pd.to_datetime(b["fecha_busqueda"])
    b["termino"] = b["termino_busqueda"].map(norm_txt)
    b = b[b["termino"] != ""].sort_values("fecha")
    b = b.drop_duplicates(["usuario", "termino"], keep="first")
    info = {"filas_total": int(total), "anonimas_descartadas": int(anon),
            "pares_usuario_termino_usados": int(len(b)), "usuarios": int(b["usuario"].nunique()),
            "terminos_distintos": int(b["termino"].nunique())}
    return b[["usuario", "termino", "fecha"]].reset_index(drop=True), info


def mapa_categorias(prod):
    """nombre normalizado de categoría -> id (los nombres vienen alineados con los ids en productos.csv)."""
    m = {}
    for ids, nombres in zip(prod["categorias"], prod["categorias_nombres"]):
        a = [x for x in ids.split("|") if x]
        n = [x for x in nombres.split("|") if x]
        if len(a) == len(n):
            for i, nom in zip(a, n):
                m[norm_txt(nom)] = i
    return m


def vectores_busqueda(terminos, nombre_a_id, tfidf, cats):
    """(Tq, Cq): una fila por término, bloque de texto y bloque de categorías."""
    Tq = normalize(tfidf.transform(terminos))
    ci = {c: i for i, c in enumerate(cats)}
    filas, cols = [], []
    for r, t in enumerate(terminos):
        cid = nombre_a_id.get(t)
        if cid in ci:
            filas.append(r)
            cols.append(ci[cid])
    Cq = sp.csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(len(terminos), max(len(cats), 1)))
    return Tq, Cq


def matriz_Q(Tq, Cq, w, n_precio):
    """Vectores de búsqueda con los mismos pesos de bloques w=(texto, categoría, precio) que X."""
    vacio = sp.csr_matrix((Tq.shape[0], n_precio))
    return normalize(sp.hstack([w[0] * Tq, w[1] * Cq, vacio]).tocsr())


def adjuntar(casos, busq, term_idx):
    """Agrega a cada caso las filas de Q de las búsquedas hechas antes del producto objetivo."""
    por_user = {u: g for u, g in busq.groupby("usuario")} if busq is not None else {}
    for c in casos:
        g = por_user.get(c["usuario"])
        if g is not None:
            g = g[g["fecha"] < c["t"]]
        c["q_rows"] = (np.array([term_idx[t] for t in g["termino"]], dtype=int)
                       if g is not None and len(g) else np.array([], dtype=int))


def score_contenido_busq(caso, cand, ctx, X=None, Q=None, wb=0.0, esquema="pesos", **kw):
    w = caso["prof_w"] if esquema == "pesos" else np.ones(len(caso["prof_w"]))
    u = sp.csr_matrix(w.reshape(1, -1)) @ X[caso["prof_rows"]]
    if wb > 0 and Q is not None and len(caso["q_rows"]):
        u = u + wb * sp.csr_matrix(np.ones((1, len(caso["q_rows"])))) @ Q[caso["q_rows"]]
    return (X[cand] @ u.T).toarray().ravel()
