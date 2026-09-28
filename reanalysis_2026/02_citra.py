"""Uji tesis 2022 — langkah 2: bangun citra untuk empat himpunan.

Cara menggambar mengikuti conv_topict di Preprocessing-new1.ipynb: kanal 0 digambar sebagai
garis, sumbu dimatikan, sumbu-y diskala otomatis per citra, figsize (75, 10). Tesis menyimpan
pada dpi 400 (30.000 x 4.000 px) lalu mengecilkan; di sini dpi 40 (3.000 x 400 px) lalu
diperkecil ke 800 x 400, ukuran masukan Tabel 3.2 tesis. Satu kanal abu-abu (tesis tanpa
sengaja memakai tiga kanal identik; lihat Verifikasi_Kode_Tesis_2022.md temuan 4).

Himpunan:
  E0   seperti tesis:  SCD = 45.000 cuplikan sebelum VF (250 Hz, 3 menit);
                       Normal = 45.000 cuplikan terakhir nsrdb (128 Hz, 5,9 menit) — di bagian
                       rekaman yang sudah tidak berisi EKG (01_periksa_data.py). Semua 18 nsrdb
                       dipakai; tesis membuang 19830 karena readlines()[:-1].
  E1   diperbaiki:     SCD sama; Normal = 3 menit dari tengah rentang yang teranotasi, diubah
                       ke 250 Hz, jadi durasi dan frekuensi sampel sama dengan SCD.
  E2   awitan digeser: SCD = 3 menit yang berakhir 60 menit sebelum VF; Normal seperti E1.
  CHF  kontrol sakit:  15 chfdb (gagal jantung berat, hidup), 3 menit dari tengah rentang
                       teranotasi, 250 Hz. Hanya potongan itu yang diunduh dari PhysioNet.

Keluaran: data/citra/<himpunan>/<kelas>_<rekaman>.png, data/citra/manifest.csv
          (himpunan, kelas, rekaman, berkas, simpangan baku dan rentang sinyal).
Pakai:    python 02_citra.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from PIL import Image
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('THESIS_DATA', os.path.join(HERE, 'data', 'physionet'))
CHF_ANN = os.path.join(DATA, 'chfdb')
CITRA = os.path.join(HERE, 'data', 'citra')
M = 45000
FS = 250
TIGA = 180 * FS
GESER = 60 * 60 * FS


def gambar(sig, berkas):
    fig, ax = plt.subplots(figsize=(75, 10))
    ax.axis('off')
    ax.plot(sig, color='#3979f0')
    tmp = berkas + '.tmp.png'
    fig.savefig(tmp, dpi=40)
    plt.close(fig)
    Image.open(tmp).convert('L').resize((800, 400), Image.LANCZOS).save(berkas)
    os.remove(tmp)


def detik(teks):
    j, m, d = (int(x) for x in teks.split('.'))
    return j * 3600 + m * 60 + d


def rentang_anotasi(jalur, ekst):
    a = wfdb.rdann(jalur, ekst)
    return int(a.sample[0]), int(a.sample[-1])


def main():
    catatan = []

    def simpan(himp, kelas, rec, sig):
        d = os.path.join(CITRA, himp)
        os.makedirs(d, exist_ok=True)
        berkas = os.path.join(d, f'{kelas}_{rec}.png')
        gambar(sig, berkas)
        catatan.append({'himpunan': himp, 'kelas': kelas, 'rekaman': rec,
                        'berkas': os.path.relpath(berkas, CITRA),
                        'std': float(np.std(sig)), 'rentang': float(np.ptp(sig))})
        print(himp, kelas, rec, flush=True)

    vf = pd.read_csv(os.path.join(DATA, 'VF Data.csv'), sep=';')
    vf = vf[~vf['VF Onset Time (elapsed)'].str.contains('VF')]
    for _, r in vf.iterrows():
        rec = r['Dat Files'].replace('.dat', '')
        jalur = os.path.join(DATA, 'sddb', rec)
        onset = int(detik(r['VF Onset Time (elapsed)']) * FS)
        sig = wfdb.rdrecord(jalur, sampfrom=onset - M, sampto=onset, channels=[0]).p_signal[:, 0]
        simpan('E0', 'SCD', rec, sig)
        simpan('E1', 'SCD', rec, sig)
        sig = wfdb.rdrecord(jalur, sampfrom=onset - GESER - TIGA, sampto=onset - GESER,
                            channels=[0]).p_signal[:, 0]
        simpan('E2', 'SCD', rec, sig)

    with open(os.path.join(DATA, 'nsrdb', 'RECORDS')) as f:
        nsr = [x.strip() for x in f.read().splitlines() if x.strip()]
    for rec in nsr:
        jalur = os.path.join(DATA, 'nsrdb', rec)
        L = wfdb.rdheader(jalur).sig_len
        sig = wfdb.rdrecord(jalur, sampfrom=L - M, sampto=L, channels=[0]).p_signal[:, 0]
        simpan('E0', 'Normal', rec, sig)
        a, b = rentang_anotasi(jalur, 'atr')
        tengah = (a + b) // 2
        n128 = 180 * 128
        sig = wfdb.rdrecord(jalur, sampfrom=tengah - n128 // 2, sampto=tengah + n128 // 2,
                            channels=[0]).p_signal[:, 0]
        sig = resample_poly(sig, 125, 64)  # 128 Hz -> 250 Hz
        simpan('E1', 'Normal', rec, sig)
        simpan('E2', 'Normal', rec, sig)

    for n in sorted(x[:-4] for x in os.listdir(CHF_ANN) if x.endswith('.hea')):
        a, b = rentang_anotasi(os.path.join(CHF_ANN, n), 'ecg')
        tengah = (a + b) // 2
        sig = wfdb.rdrecord(n, pn_dir='chfdb', sampfrom=tengah - TIGA // 2, sampto=tengah + TIGA // 2,
                            channels=[0]).p_signal[:, 0]
        simpan('CHF', 'CHF', n, sig)

    pd.DataFrame(catatan).to_csv(os.path.join(CITRA, 'manifest.csv'), index=False)
    print(f'{len(catatan)} citra')


if __name__ == '__main__':
    main()
