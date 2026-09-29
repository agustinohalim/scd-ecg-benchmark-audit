"""Uji tesis 2022 — langkah 6: kontrol pita frekuensi untuk E1.

Normal E1 = nsrdb 128 Hz diubah ke 250 Hz (pita <= 64 Hz). SCD dan CHF asli 250 Hz (pita <= 125 Hz).
Di sini SCD dan CHF dilewatkan 250 -> 128 -> 250 Hz, jalur yang sama dengan Normal, lalu 2D-CNN
"benar" dilatih ulang dengan lipatan, benih, dan epoch yang sama dengan 03_latih.py.
Citra Normal E1 dipakai apa adanya. Data SCD/CHF dibaca langsung dari PhysioNet (pn_dir).

Keluaran: data/citra_pita/, keluaran/hasil_E1r_cnn_benar.csv, keluaran/oof_E1r_cnn_benar.csv
Pakai:    python 06_kontrol_pita.py "<jalur VF Data.csv>"
"""
import importlib.util
import os
import shutil
import sys

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.abspath(__file__))
KINI = REPO
CITRA = os.path.join(REPO, 'data', 'citra_pita')


def modul(nama, berkas):
    s = importlib.util.spec_from_file_location(nama, os.path.join(REPO, berkas))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


citra = modul('citra', '02_citra.py')
latih = modul('latih', '03_latih.py')
FS, M, TIGA = 250, 45000, 180 * 250


def pita(sig):
    return resample_poly(resample_poly(sig, 64, 125), 125, 64)


def main():
    os.makedirs(os.path.join(CITRA, 'E1r'), exist_ok=True)
    os.makedirs(os.path.join(CITRA, 'CHF'), exist_ok=True)
    man0 = pd.read_csv(os.path.join(REPO, 'data', 'citra', 'manifest.csv'))
    baris = []
    for _, r in man0[(man0.himpunan == 'E1') & (man0.kelas == 'Normal')].iterrows():
        tuju = os.path.join('E1r', os.path.basename(r.berkas))
        shutil.copy(os.path.join(REPO, 'data', 'citra', r.berkas), os.path.join(CITRA, tuju))
        baris.append({'himpunan': 'E1r', 'kelas': 'Normal', 'rekaman': r.rekaman, 'berkas': tuju})

    vf = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(latih.HERE, 'data', 'physionet', 'VF Data.csv'),
                     sep=';')
    vf = vf[~vf['VF Onset Time (elapsed)'].str.contains('VF')]
    for _, r in vf.iterrows():
        rec = r['Dat Files'].replace('.dat', '')
        onset = int(citra.detik(r['VF Onset Time (elapsed)']) * FS)
        sig = wfdb.rdrecord(rec, pn_dir='sddb', sampfrom=onset - M, sampto=onset, channels=[0]).p_signal[:, 0]
        tuju = os.path.join('E1r', f'SCD_{rec}.png')
        citra.gambar(pita(sig), os.path.join(CITRA, tuju))
        baris.append({'himpunan': 'E1r', 'kelas': 'SCD', 'rekaman': rec, 'berkas': tuju})
        print('SCD', rec, flush=True)

    chf = man0[man0.himpunan == 'CHF']
    for n in chf.rekaman.astype(str):
        n = n.zfill(3) if not n.startswith('chf') else n
        a = wfdb.rdann(n, 'ecg', pn_dir='chfdb')
        tengah = (int(a.sample[0]) + int(a.sample[-1])) // 2
        sig = wfdb.rdrecord(n, pn_dir='chfdb', sampfrom=tengah - TIGA // 2, sampto=tengah + TIGA // 2,
                            channels=[0]).p_signal[:, 0]
        tuju = os.path.join('CHF', f'CHF_{n}.png')
        citra.gambar(pita(sig), os.path.join(CITRA, tuju))
        baris.append({'himpunan': 'CHF', 'kelas': 'CHF', 'rekaman': n, 'berkas': tuju})
        print('CHF', n, flush=True)

    pd.DataFrame(baris).to_csv(os.path.join(CITRA, 'manifest.csv'), index=False)
    latih.CITRA = CITRA
    latih.OUT = os.path.join(KINI, 'keluaran')
    return latih.main(['E1r', 'cnn', 'benar', '--chf'])


if __name__ == '__main__':
    sys.exit(main())
