"""Uji tesis 2022 — langkah 4: garis dasar remeh untuk setiap himpunan.

Bila satu angka yang tidak tahu apa-apa tentang jantung — simpangan baku sinyal di jendela —
memisahkan kelas setara jaringan saraf, akurasi jaringan itu tidak bisa dibaca sebagai bukti
bahwa ia mengenali SCD.

Regresi logistik atas log simpangan baku (satu ciri), lipatan dan ulangan sama dengan
03_latih.py (5 x 3, benih 2026), AUC dan akurasi dari prediksi di luar lipatan.

Keluaran: keluaran/garis_dasar.csv
Pakai:    python 04_garis_dasar.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    man = pd.read_csv(os.path.join(HERE, 'data', 'citra', 'manifest.csv'))
    baris = []
    for himp in ['E0', 'E1', 'E2']:
        d = man[man.himpunan == himp]
        hilang = d[d['std'].isna()]
        if len(hilang):  # sddb 52 punya cuplikan kosong di jendela E2; citranya bercelah
            print(f'{himp}: dilewati karena sinyal kosong: {", ".join(hilang.rekaman.astype(str))}')
        d = d.dropna(subset=['std']).reset_index(drop=True)
        x = np.log(d[['std']].values)
        y = (d.kelas == 'SCD').astype(int).values
        oof = np.zeros((3, len(d)))
        for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=2026)
                                     .split(x, y)):
            oof[k // 5, te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
        p = oof.mean(0)
        baris.append({'himpunan': himp, 'ciri': 'log simpangan baku',
                      'auc': roc_auc_score(y, p), 'akurasi': ((p > 0.5) == y).mean(),
                      'std_median_scd': d[y == 1]['std'].median(), 'std_median_normal': d[y == 0]['std'].median()})
    h = pd.DataFrame(baris)
    h.to_csv(os.path.join(HERE, 'keluaran', 'garis_dasar.csv'), index=False)
    print(h.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
