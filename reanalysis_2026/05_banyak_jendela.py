"""Uji tesis 2022 — langkah 5: enam jendela per pasien, bukan satu.

Langkah 3 memakai satu citra per rekaman (38 sampel), jadi selangnya lebar. Di sini setiap
pasien menyumbang enam jendela 3 menit, 250 Hz, digambar seperti 02_citra.py. Rancangan
ditetapkan sebelum dijalankan:

  SCD     6 jendela yang berakhir 0, 10, 20, 30, 40, 50 menit sebelum awitan VF
  Normal  6 jendela berjarak sama di rentang teranotasi nsrdb (10 %-90 %), diubah ke 250 Hz
  CHF     6 jendela berjarak sama di rentang teranotasi chfdb (10 %-90 %); hanya dinilai

Jendela dengan > 1 % cuplikan kosong dilewati. Model 2D-CNN Tabel 3.2, pengaturan "benar" dari
03_latih.py, 15 epoch (jumlah langkah gradien sekitar tiga kali langkah 3). Lipatan tingkat
PASIEN: semua jendela seorang pasien berada di sisi yang sama. 5 lipatan x 2 ulangan, benih 2026.

Dilaporkan: AUC tingkat jendela; AUC tingkat pasien (peluang rata-rata jendela), selang 95 %
bootstrap pasien; AUC per horizon (jendela SCD pada horizon itu lawan semua jendela Normal);
proporsi pasien CHF yang peluang rata-ratanya > 0,5.

Keluaran: data/citra/B/..., keluaran/banyak_jendela.csv (peluang per jendela per ulangan),
          keluaran/hasil_banyak_jendela.csv
Pakai:    python 05_banyak_jendela.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))


def modul(nama, berkas):
    s = importlib.util.spec_from_file_location(nama, os.path.join(HERE, berkas))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


citra = modul('citra', '02_citra.py')
latih = modul('latih', '03_latih.py')
DATA, CHF_ANN, FS = citra.DATA, citra.CHF_ANN, citra.FS
TIGA = 180 * FS
B = os.path.join(HERE, 'data', 'citra', 'B')
OUT = os.path.join(HERE, 'keluaran')
HORIZON = [0, 10, 20, 30, 40, 50]
BENIH, LIPAT, ULANG, EPOCH = 2026, 5, 2, 15


def titik(a, b, n=6):
    return [int(a + (b - a) * (0.1 + 0.8 * i / (n - 1))) for i in range(n)]


def bangun():
    baris = []

    def simpan(kelas, rec, j, sig):
        if np.mean(np.isnan(sig)) > 0.01:
            return
        berkas = os.path.join(B, f'{kelas}_{rec}_{j}.png')
        if not os.path.exists(berkas):
            citra.gambar(sig, berkas)
        baris.append({'kelas': kelas, 'rekaman': str(rec), 'jendela': j, 'berkas': berkas})

    os.makedirs(B, exist_ok=True)
    vf = pd.read_csv(os.path.join(DATA, 'VF Data.csv'), sep=';')
    vf = vf[~vf['VF Onset Time (elapsed)'].str.contains('VF')]
    for _, r in vf.iterrows():
        rec = r['Dat Files'].replace('.dat', '')
        onset = int(citra.detik(r['VF Onset Time (elapsed)']) * FS)
        for h in HORIZON:
            akhir = onset - h * 60 * FS
            sig = wfdb.rdrecord(os.path.join(DATA, 'sddb', rec), sampfrom=akhir - TIGA, sampto=akhir,
                                channels=[0]).p_signal[:, 0]
            simpan('SCD', rec, h, sig)
    with open(os.path.join(DATA, 'nsrdb', 'RECORDS')) as f:
        nsr = [x.strip() for x in f.read().splitlines() if x.strip()]
    for rec in nsr:
        jalur = os.path.join(DATA, 'nsrdb', rec)
        a, b = citra.rentang_anotasi(jalur, 'atr')
        for j, t in enumerate(titik(a, b)):
            sig = wfdb.rdrecord(jalur, sampfrom=t, sampto=t + 180 * 128, channels=[0]).p_signal[:, 0]
            simpan('Normal', rec, j, resample_poly(sig, 125, 64))
    for n in sorted(x[:-4] for x in os.listdir(CHF_ANN) if x.endswith('.hea')):
        a, b = citra.rentang_anotasi(os.path.join(CHF_ANN, n), 'ecg')
        for j, t in enumerate(titik(a, b)):
            sig = wfdb.rdrecord(n, pn_dir='chfdb', sampfrom=t, sampto=t + TIGA, channels=[0]).p_signal[:, 0]
            simpan('CHF', n, j, sig)
        print('CHF', n, flush=True)
    return pd.DataFrame(baris)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = bangun()
    print(d.groupby('kelas').size().to_string(), flush=True)
    latihd = d[d.kelas != 'CHF'].reset_index(drop=True)
    chf = d[d.kelas == 'CHF'].reset_index(drop=True)
    muat = lambda bs: torch.tensor(np.stack([latih.muat(os.path.relpath(b, latih.CITRA)) for b in bs]))
    x, xc = muat(latihd.berkas), muat(chf.berkas)
    y_np = (latihd.kelas == 'SCD').astype(int).values
    y = torch.tensor(y_np)

    pasien = latihd.drop_duplicates('rekaman')[['rekaman', 'kelas']].reset_index(drop=True)
    yp = (pasien.kelas == 'SCD').astype(int).values
    latih.EPOCH = EPOCH
    peluang = np.zeros((ULANG, len(latihd)))
    pchf = []
    mulai = time.time()
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=LIPAT, n_repeats=ULANG, random_state=BENIH)
                                 .split(np.zeros(len(pasien)), yp)):
        itr = np.where(latihd.rekaman.isin(pasien.rekaman[tr]))[0]
        ite = np.where(latihd.rekaman.isin(pasien.rekaman[te]))[0]
        torch.manual_seed(BENIH + k)
        m = latih.buat('cnn')
        loss = latih.latih(m, x[itr], y[itr], 'benar')
        peluang[k // LIPAT, ite] = latih.peluang(m, x[ite], 'benar')
        pchf.append(latih.peluang(m, xc, 'benar'))
        print(f'lipatan {k + 1}/{LIPAT * ULANG}: galat akhir {loss:.4f}, {time.time() - mulai:.0f} s', flush=True)

    latihd[[f'u{r + 1}' for r in range(ULANG)]] = peluang.T
    latihd.to_csv(os.path.join(OUT, 'banyak_jendela.csv'), index=False)
    p = peluang.mean(0)
    hasil = [{'ukuran': 'AUC tingkat jendela', 'nilai': roc_auc_score(y_np, p)}]
    per = latihd.assign(p=p).groupby('rekaman').agg(p=('p', 'mean'), kelas=('kelas', 'first')).reset_index()
    yy = (per.kelas == 'SCD').astype(int).values
    hasil.append({'ukuran': 'AUC tingkat pasien', 'nilai': roc_auc_score(yy, per.p)})
    rng = np.random.default_rng(BENIH)
    boot = [roc_auc_score(yy[i], per.p.values[i]) for i in (rng.integers(0, len(per), len(per)) for _ in range(2000))
            if len(set(yy[i])) > 1]
    hasil += [{'ukuran': 'AUC tingkat pasien, batas bawah 95 %', 'nilai': np.percentile(boot, 2.5)},
              {'ukuran': 'AUC tingkat pasien, batas atas 95 %', 'nilai': np.percentile(boot, 97.5)},
              {'ukuran': 'Akurasi tingkat pasien (ambang 0,5)', 'nilai': float(((per.p > 0.5) == yy).mean())}]
    nor = latihd.kelas == 'Normal'
    for h in HORIZON:
        pilih = nor | ((latihd.kelas == 'SCD') & (latihd.jendela == h))
        hasil.append({'ukuran': f'AUC horizon {h} menit sebelum VF', 'nilai': roc_auc_score(y_np[pilih], p[pilih])})
    pc = pd.DataFrame({'rekaman': chf.rekaman, 'p': np.mean(pchf, 0)}).groupby('rekaman').p.mean()
    hasil += [{'ukuran': 'CHF: proporsi pasien dinyatakan SCD', 'nilai': float((pc > 0.5).mean())},
              {'ukuran': 'CHF: peluang SCD median per pasien', 'nilai': float(pc.median())},
              {'ukuran': 'Normal: peluang SCD median per pasien', 'nilai': float(per[per.kelas == 'Normal'].p.median())},
              {'ukuran': 'SCD: peluang SCD median per pasien', 'nilai': float(per[per.kelas == 'SCD'].p.median())},
              {'ukuran': 'jendela SCD / Normal / CHF', 'nilai': f"{(latihd.kelas == 'SCD').sum()} / {nor.sum()} / {len(chf)}"}]
    h = pd.DataFrame(hasil)
    h.to_csv(os.path.join(OUT, 'hasil_banyak_jendela.csv'), index=False)
    print(h.to_string(index=False))


if __name__ == '__main__':
    main()
