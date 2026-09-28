"""Uji tesis 2022 — langkah 1: periksa data yang dipakai tesis, sebelum melatih apa pun.

Tesis (dan Preprocessing-new1.ipynb) mengambil:
  SCD    = m = 45.000 cuplikan tepat sebelum awitan VF, rekaman sddb (20 rekaman ber-VF)
  Normal = m = 45.000 cuplikan terakhir rekaman nsrdb
dengan m = (1000*180)/4, yaitu 180 detik bila 1 cuplikan = 4 ms.

Yang diperiksa di sini, per rekaman, dari header dan anotasi PhysioNet:
  - frekuensi sampel yang sebenarnya, dan berapa detik yang dicakup 45.000 cuplikan;
  - apakah awitan VF di `VF Data.csv` jatuh di dalam rekaman;
  - isi irama jendela SCD: label irama (aux_note) dan proporsi denyut ventrikel (V) pada
    anotasi .ari (ada untuk semua rekaman sddb; .atr teraudit hanya untuk sebagian);
  - denyut per menit di jendela tesis dan di jendela 3 menit yang setara;
  - rekaman nsrdb yang dibuang `readlines()[:-1]`.

Keluaran: keluaran/periksa_data.csv dan ringkasan di layar.
Pakai:    python 01_periksa_data.py
"""
import os

import numpy as np
import pandas as pd
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('THESIS_DATA', os.path.join(HERE, 'data', 'physionet'))
OUT = os.path.join(HERE, 'keluaran')
M = int((1000 * 180) / 4)  # 45.000 cuplikan, seperti di notebook
BUKAN_DENYUT = set('+~|"[]!x()pt^u`\'=s*@')  # anotasi yang bukan kompleks QRS


def detik(teks):
    j, m, d = (int(x) for x in teks.split('.'))
    return j * 3600 + m * 60 + d


def denyut(ann, a, b):
    """Simbol denyut dalam [a, b) cuplikan."""
    s = [sym for smp, sym in zip(ann.sample, ann.symbol) if a <= smp < b and sym not in BUKAN_DENYUT]
    return s


def irama_di(ann, a, b):
    """Label irama yang berlaku di [a, b): yang terakhir sebelum a, ditambah yang berganti di dalamnya."""
    label, sebelum = [], None
    for smp, aux in zip(ann.sample, ann.aux_note):
        aux = aux.strip('\x00 ').strip()
        if not aux.startswith('('):
            continue
        if smp < a:
            sebelum = aux
        elif smp < b:
            label.append(aux)
    return ([sebelum] if sebelum else []) + label


def main():
    os.makedirs(OUT, exist_ok=True)
    baris = []

    vf = pd.read_csv(os.path.join(DATA, 'VF Data.csv'), sep=';')
    vf = vf[~vf['VF Onset Time (elapsed)'].str.contains('VF')]
    for _, r in vf.iterrows():
        rec = r['Dat Files'].replace('.dat', '')
        jalur = os.path.join(DATA, 'sddb', rec)
        h = wfdb.rdheader(jalur)
        onset = int(detik(r['VF Onset Time (elapsed)']) * h.fs)
        a = onset - M
        ann = wfdb.rdann(jalur, 'ari')
        s = denyut(ann, a, onset)
        teraudit = os.path.exists(jalur + '.atr')
        s_atr = denyut(wfdb.rdann(jalur, 'atr'), a, onset) if teraudit else None
        baris.append({
            'kelas': 'SCD', 'rekaman': rec, 'fs': h.fs, 'lead': h.sig_name[0],
            'durasi_jam': round(h.sig_len / h.fs / 3600, 2),
            'onset_di_dalam': 0 < onset <= h.sig_len,
            'detik_jendela_tesis': round(M / h.fs, 1),
            'bpm_jendela_tesis': round(len(s) / (M / h.fs) * 60, 1),
            'proporsi_V': round(sum(x == 'V' for x in s) / max(len(s), 1), 3),
            'proporsi_V_atr': (round(sum(x == 'V' for x in s_atr) / max(len(s_atr), 1), 3) if s_atr else ''),
            'irama': ' '.join(irama_di(ann, a, onset)),
        })

    with open(os.path.join(DATA, 'nsrdb', 'RECORDS')) as f:
        semua = [x.strip() for x in f.read().splitlines() if x.strip()]
    with open(os.path.join(DATA, 'nsrdb', 'RECORDS')) as f:
        versi_tesis = [x[:-1] for x in f.readlines()[:-1]]
    dibuang = sorted(set(semua) - set(versi_tesis))
    for rec in semua:
        jalur = os.path.join(DATA, 'nsrdb', rec)
        h = wfdb.rdheader(jalur)
        ann = wfdb.rdann(jalur, 'atr')
        akhir = h.sig_len
        s_tesis = denyut(ann, akhir - M, akhir)
        tiga = int(180 * h.fs)
        s_3 = denyut(ann, akhir - tiga, akhir)
        sig = wfdb.rdrecord(jalur, sampfrom=akhir - M, sampto=akhir, channels=[0]).p_signal[:, 0]
        tengah = wfdb.rdrecord(jalur, sampfrom=akhir // 2, sampto=akhir // 2 + M, channels=[0]).p_signal[:, 0]
        baris.append({
            'kelas': 'Normal', 'rekaman': rec, 'fs': h.fs, 'lead': h.sig_name[0],
            'durasi_jam': round(h.sig_len / h.fs / 3600, 2), 'onset_di_dalam': '',
            'detik_jendela_tesis': round(M / h.fs, 1),
            'bpm_jendela_tesis': round(len(s_tesis) / (M / h.fs) * 60, 1),
            'bpm_3_menit': round(len(s_3) / 3, 1),
            'proporsi_V': round(sum(x == 'V' for x in s_tesis) / max(len(s_tesis), 1), 3),
            'irama': ' '.join(irama_di(ann, akhir - M, akhir)),
            'datar': round(float(np.mean(np.abs(np.diff(sig)) < 1e-9)), 3),
            'menit_sesudah_anotasi_terakhir': round((akhir - ann.sample[-1]) / h.fs / 60, 1),
            'rentang_mV_jendela_tesis': round(float(np.ptp(sig)), 2),
            'std_mV_jendela_tesis': round(float(np.std(sig)), 3),
            'std_mV_tengah_rekaman': round(float(np.std(tengah)), 3),
            'dibuang_tesis': rec in dibuang,
        })

    df = pd.DataFrame(baris)
    df.to_csv(os.path.join(OUT, 'periksa_data.csv'), index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    print(df.to_string(index=False))
    scd, nor = df[df.kelas == 'SCD'], df[df.kelas == 'Normal']
    print(f"\nSCD: {len(scd)} rekaman, fs {sorted(scd.fs.unique())}, jendela tesis "
          f"{scd.detik_jendela_tesis.iloc[0]} s, bpm median {scd.bpm_jendela_tesis.median()}")
    print(f"Normal: {len(nor)} rekaman, fs {sorted(nor.fs.unique())}, jendela tesis "
          f"{nor.detik_jendela_tesis.iloc[0]} s, bpm median {nor.bpm_jendela_tesis.median()} "
          f"(3 menit: {nor.bpm_3_menit.median()})")
    print(f"Normal: menit sesudah anotasi terakhir {nor.menit_sesudah_anotasi_terakhir.min()}–"
          f"{nor.menit_sesudah_anotasi_terakhir.max()}; std jendela tesis median {nor.std_mV_jendela_tesis.median()} "
          f"mV lawan tengah rekaman {nor.std_mV_tengah_rekaman.median()} mV; "
          f"jendela tesis tanpa denyut teranotasi: {(nor.bpm_jendela_tesis == 0).sum()} dari {len(nor)}")
    print(f"Dibuang readlines()[:-1]: {dibuang}")
    print(f"SCD dengan denyut V >= 10 % di jendela: {(scd.proporsi_V >= 0.10).sum()} dari {len(scd)}")


if __name__ == '__main__':
    main()
