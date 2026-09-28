"""Uji tesis 2022 — langkah 3: latih dan uji model tesis, versi diperbaiki dan versi tesis.

Model (PyTorch; tesis memakai Keras, TensorFlow tidak terpasang di mesin ini):
  cnn          Tabel 3.2 tesis: Conv(32)-Pool-Conv(32)-Pool-Conv(64)-Pool-Dropout(0,4)-Flatten-
               Dense(128)-Dense(2), masukan 400 x 800, satu kanal.
  lstm         batang konvolusi yang sama, lalu BiLSTM(128) yang berjalan sepanjang LEBAR citra
               (sumbu waktu), Dense(256), Dense(2). Perbaikan atas Modeling-new.ipynb.
  lstm_tesis   seperti Modeling-new.ipynb: ReshapeLayer membuat TINGGI citra (amplitudo)
               menjadi urutan LSTM. Keras memakai activation='relu' di LSTM; PyTorch hanya tanh.

Pengaturan latih:
  benar        Adam lr 1e-4, galat entropi silang atas logit, 30 epoch tetap (tidak ada himpunan
               uji yang dipantau selama latih), batch 4.
  tesis        seperti sel 25 Modeling-new.ipynb: Adam lr 1e-6 dan softmax ganda (galat entropi
               silang dikenakan pada keluaran softmax). 30 epoch, bukan 200, karena biaya CPU.

Evaluasi: satu citra per rekaman, jadi lipatan tingkat rekaman = tingkat pasien. 5 lipatan
berstrata diulang 3 kali (benih 2026). Per ulangan: AUC, akurasi, sensitivitas, spesifisitas
pada ambang 0,5 dari prediksi di luar lipatan. Gabungan: peluang dirata-rata per pasien atas
ulangan, AUC dengan selang 95 % bootstrap pasien (2.000 kali). --chf: setiap model lipatan juga
menilai 15 citra chfdb; dilaporkan proporsi yang dinyatakan SCD (peluang rata-rata > 0,5).

Keluaran: keluaran/hasil_<himpunan>_<model>_<pengaturan>.csv (per ulangan + gabungan),
          keluaran/oof_<...>.csv (peluang per pasien per ulangan).
  --balik      membalik citra (latar putih 1 menjadi 0) sebelum masuk model.

Pakai:    python 03_latih.py E1 cnn benar [--chf] [--balik]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
CITRA = os.path.join(HERE, 'data', 'citra')
OUT = os.path.join(HERE, 'keluaran')
BENIH, LIPAT, ULANG, EPOCH, BATCH = 2026, 5, 3, 30, 4
torch.set_num_threads(12)


class Batang(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.4))

    def forward(self, x):
        return self.f(x)  # (B, 64, 48, 98) untuk masukan 400 x 800


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = Batang()
        self.k = nn.Sequential(nn.Flatten(), nn.Linear(64 * 48 * 98, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.k(self.b(x))


class CNNLSTM(nn.Module):
    def __init__(self, sumbu_waktu=True):
        super().__init__()
        self.b = Batang()
        self.waktu = sumbu_waktu
        masuk = 64 * 48 if sumbu_waktu else 64 * 98
        self.l = nn.LSTM(masuk, 128, batch_first=True, bidirectional=True)
        self.k = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        z = self.b(x)  # (B, C, H, W)
        if self.waktu:
            z = z.permute(0, 3, 1, 2).flatten(2)  # urutan sepanjang W (waktu)
        else:
            z = z.permute(0, 2, 1, 3).flatten(2)  # urutan sepanjang H (amplitudo), seperti tesis
        _, (h, _) = self.l(z)
        return self.k(torch.cat([h[0], h[1]], dim=1))


def buat(model):
    return {'cnn': CNN, 'lstm': lambda: CNNLSTM(True), 'lstm_tesis': lambda: CNNLSTM(False)}[model]()


def muat(berkas):
    return np.asarray(Image.open(os.path.join(CITRA, berkas)), dtype=np.float32)[None] / 255.0


def latih(model, x, y, pengaturan):
    lr = 1e-4 if pengaturan == 'benar' else 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    galat = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCH):
        urut = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            j = urut[i:i + BATCH]
            keluar = model(x[j])
            if pengaturan == 'tesis':
                keluar = torch.softmax(keluar, 1)  # softmax di lapisan, lalu galat mengira logit
            loss = galat(keluar, y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.detach())


@torch.no_grad()
def peluang(model, x, pengaturan):
    model.eval()
    p = []
    for i in range(0, len(x), 8):
        keluar = model(x[i:i + 8])
        if pengaturan == 'tesis':
            keluar = torch.softmax(keluar, 1)
        p.append(torch.softmax(keluar, 1)[:, 1])
    return torch.cat(p).numpy()


def ukur(y, p):
    t = (p > 0.5).astype(int)
    tp, tn = int(((t == 1) & (y == 1)).sum()), int(((t == 0) & (y == 0)).sum())
    return {'auc': roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
            'akurasi': (tp + tn) / len(y), 'sensitivitas': tp / max((y == 1).sum(), 1),
            'spesifisitas': tn / max((y == 0).sum(), 1)}


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    himp, model_nama, pengaturan = argv[:3]
    pakai_chf = '--chf' in argv
    balik = '--balik' in argv  # latar 0, jejak 1; tanpa ini LSTM tidak belajar (lihat Hasil)
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(CITRA, 'manifest.csv'))
    d = man[man.himpunan == himp].reset_index(drop=True)
    x = torch.tensor(np.stack([muat(b) for b in d.berkas]))
    if balik:
        x = 1 - x
    y_np = (d.kelas == 'SCD').astype(int).values
    y = torch.tensor(y_np)
    xc = None
    if pakai_chf:
        c = man[man.himpunan == 'CHF'].reset_index(drop=True)
        xc = torch.tensor(np.stack([muat(b) for b in c.berkas]))
        if balik:
            xc = 1 - xc
        pc = []

    rskf = RepeatedStratifiedKFold(n_splits=LIPAT, n_repeats=ULANG, random_state=BENIH)
    oof = np.zeros((ULANG, len(d)))
    mulai = time.time()
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(d)), y_np)):
        torch.manual_seed(BENIH + k)
        m = buat(model_nama)
        loss = latih(m, x[tr], y[tr], pengaturan)
        oof[k // LIPAT, te] = peluang(m, x[te], pengaturan)
        if pakai_chf:
            pc.append(peluang(m, xc, pengaturan))
        print(f'lipatan {k + 1}/{LIPAT * ULANG}: galat akhir {loss:.4f}, '
              f'{time.time() - mulai:.0f} s', flush=True)

    baris = [{'ulangan': r + 1, **ukur(y_np, oof[r])} for r in range(ULANG)]
    rata = oof.mean(0)
    gab = {'ulangan': 'gabungan', **ukur(y_np, rata)}
    rng = np.random.default_rng(BENIH)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        if len(set(y_np[i])) > 1:
            boot.append(roc_auc_score(y_np[i], rata[i]))
    gab['auc_bawah'], gab['auc_atas'] = np.percentile(boot, [2.5, 97.5])
    if pakai_chf:
        pchf = np.mean(pc, 0)
        gab['chf_dinyatakan_scd'] = float((pchf > 0.5).mean())
        gab['chf_peluang_median'] = float(np.median(pchf))
        gab['normal_peluang_median'] = float(np.median(rata[y_np == 0]))
        gab['scd_peluang_median'] = float(np.median(rata[y_np == 1]))
    baris.append(gab)
    nama = f'{himp}_{model_nama}_{pengaturan}' + ('_balik' if balik else '')
    hasil = pd.DataFrame(baris)
    hasil.to_csv(os.path.join(OUT, f'hasil_{nama}.csv'), index=False)
    pd.DataFrame(oof.T, columns=[f'u{r + 1}' for r in range(ULANG)]).assign(
        rekaman=d.rekaman, kelas=d.kelas).to_csv(os.path.join(OUT, f'oof_{nama}.csv'), index=False)
    print(hasil.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
