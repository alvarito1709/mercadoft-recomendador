#!/usr/bin/env python3
"""
Variante "con señal de búsqueda" del modelo de contenido (se corre DESPUÉS de eda_recomendador.py).

Idea: cada término buscado por el usuario (antes del producto objetivo) se convierte en un vector del mismo
espacio que los productos (TF-IDF del término + categoría si el término es el nombre de una categoría) y se
suma al perfil con un peso w_b. w_b = 0 es el modelo actual (sin búsqueda). Se usan los MISMOS casos que
modelo_eval.py, así que la comparación es pareada.

Solo entran búsquedas con usuario_id (las anónimas no se pueden atar a un perfil) de usuarios que están en
eventos.csv, y de cada (usuario, término) cuenta una sola vez (la primera, para no inflar repeticiones).

Uso:
  uv run python eval_busqueda.py --data-dir data --in out --out out
Requiere modelo_eval.py en la misma carpeta y el salt de out/.salt (o variable RECO_SALT).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mft import busqueda as bq
import modelo_eval as me

PESOS_BUSQUEDA = me.PESOS_BUSQUEDA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--in", dest="inp", default="out")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    din, dout, ddata = Path(a.inp), Path(a.out), Path(a.data_dir)
    rng = np.random.default_rng(me.SEMILLA)

    ev, prod = me.cargar(din)
    pares = me.armar_pares(ev)
    idx = {int(p): i for i, p in enumerate(prod["producto_id"])}
    eleg_rows = np.where(prod["elegible"].to_numpy())[0]
    eleg_ids = set(prod.loc[prod["elegible"], "producto_id"].astype(int))
    carga = prod["fecha_carga"].to_numpy(dtype="datetime64[ns]")
    ctx = me.Contexto(pares, idx, len(prod))
    T, C, P, tfidf, cats = me.bloques_features(prod)

    busq, info = bq.cargar_busquedas(ddata, dout, set(ev["usuario"]))
    if busq is None:
        raise SystemExit(f"No se puede usar la señal de búsqueda: {info}")
    terminos = sorted(busq["termino"].unique())
    term_idx = {t: i for i, t in enumerate(terminos)}
    Tq, Cq = bq.vectores_busqueda(terminos, bq.mapa_categorias(prod), tfidf, cats)
    info["terminos_que_son_categoria"] = int((np.asarray(Cq.sum(axis=1)).ravel() > 0).sum())
    info["terminos_sin_vocabulario_ni_categoria"] = int(
        ((np.asarray(Tq.sum(axis=1)).ravel() == 0) & (np.asarray(Cq.sum(axis=1)).ravel() == 0)).sum())
    print("Búsquedas:", info)

    casos = {m: me.construir_casos(pares, idx, eleg_ids, m) for m in ("A_val", "A_test", "B")}
    for m in casos:
        bq.adjuntar(casos[m], busq, term_idx)
    info["casos"] = {m: {"n": len(v), "con_busqueda_previa": int(sum(len(c["q_rows"]) > 0 for c in v))}
                     for m, v in casos.items()}
    print("Casos:", info["casos"])

    filas = []
    dfs = {}
    for w in me.GRID_BLOQUES:
        X = me.armar_X(T, C, P, w)
        Q = bq.matriz_Q(Tq, Cq, w, P.shape[1])
        for wb in PESOS_BUSQUEDA:
            for proto in ("A_val", "A_test", "B"):
                df = me.evaluar(casos[proto], bq.score_contenido_busq, ctx, eleg_rows, carga, rng, X=X, Q=Q, wb=wb)
                dfs[(w, wb, proto)] = df
        for wb in PESOS_BUSQUEDA:
            fila = {"texto": w[0], "categoria": w[1], "precio": w[2], "peso_busqueda": wb}
            for proto in ("A_val", "A_test", "B"):
                df = dfs[(w, wb, proto)]
                base = dfs[(w, 0, proto)]
                m = np.array([len(c["q_rows"]) > 0 for c in casos[proto]])
                fila[f"{proto}_hit10"] = round(float(df["hit10"].mean()), 4)
                fila[f"{proto}_ndcg10"] = round(float(df["ndcg10"].mean()), 4)
                fila[f"{proto}_dif_ndcg_vs_sin"] = round(float((df["ndcg10"] - base["ndcg10"]).mean()), 4)
                if m.any():
                    fila[f"{proto}_solo_con_busqueda_n"] = int(m.sum())
                    fila[f"{proto}_solo_con_busqueda_ndcg"] = round(float(df["ndcg10"][m].mean()), 4)
                    fila[f"{proto}_solo_con_busqueda_ndcg_sin"] = round(float(base["ndcg10"][m].mean()), 4)
                    d = (df["ndcg10"][m] - base["ndcg10"][m]).to_numpy()
                    fila[f"{proto}_solo_con_busqueda_dif_ic95"] = me.ic(d, rng) if wb > 0 else None
            filas.append(fila)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(dout / "resultados_busqueda.csv", index=False)
    with open(dout / "resultados_busqueda.json", "w", encoding="utf-8") as f:
        json.dump({"info": info, "filas": filas}, f, indent=2, ensure_ascii=False)

    cols = ["texto", "categoria", "precio", "peso_busqueda", "A_val_ndcg10", "A_val_hit10",
            "A_test_ndcg10", "A_test_hit10", "B_ndcg10", "A_test_solo_con_busqueda_n"]
    print("\n", tabla[cols].to_string(index=False))
    print("\nListo: resultados_busqueda.csv / .json en", dout.resolve())


if __name__ == "__main__":
    main()
