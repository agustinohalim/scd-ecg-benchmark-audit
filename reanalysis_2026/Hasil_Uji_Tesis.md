# Hasil Uji Tesis 2022 — program diperbaiki, data diperiksa

Dijalankan 28–29 September 2026. Skrip `01_periksa_data.py`, `02_citra.py`, `03_latih.py`,
`04_garis_dasar.py`. Tabel di `keluaran/periksa_data.csv`, `keluaran/garis_dasar.csv`,
`keluaran/hasil_*.csv`; log `keluaran/latih.log` dan `keluaran/latih_balik.log`. Citra di
`data/citra/` (tidak masuk repo). Berkas ini satu-satunya sumber angka untuk uji ini.

**Pertanyaan.** Apakah hasil tesis (2D-CNN 96,67 %, LSTM "tidak cocok dengan data") bertahan bila
program diperbaiki, dan apa yang sebenarnya dipelajari model?

**Rancangan singkat.** Model Tabel 3.2 tesis (2D-CNN) dan CNN-LSTM, ditulis ulang di PyTorch
(TensorFlow tidak terpasang), masukan 400 x 800 satu kanal. Pengaturan "benar": Adam 1e-4,
entropi silang atas logit, 30 epoch tetap, himpunan uji tidak dipantau selama latih. Satu citra
per rekaman, 5 lipatan berstrata diulang 3 kali (benih 2026), jadi pembagian tingkat pasien.
AUC dari peluang yang dirata-rata per pasien, selang 95 % bootstrap pasien.

---

## 1. Kelas "Normal" tesis tidak berisi detak jantung

`Preprocessing-new1.ipynb` mengambil **45.000 cuplikan terakhir** setiap rekaman `nsrdb`. Di
bagian itu rekaman sudah tidak berisi EKG:

| Ukuran (18 rekaman `nsrdb`) | Nilai |
|---|---|
| Menit antara anotasi denyut terakhir dan akhir rekaman | 63,8 – 297,0 |
| Jendela tesis tanpa satu pun denyut teranotasi | **18 dari 18** |
| Simpangan baku sinyal, median, jendela tesis | 0,065 mV |
| Simpangan baku sinyal, median, tengah rekaman | 0,29 mV |

Diperiksa dengan mata pada 7 rekaman (`keluaran/cek_jendela.png`, `keluaran/cek_ekor_nsrdb.png`):
garis datar bertangga kuantisasi ±0,05 mV, tanpa kompleks QRS — elektroda sudah terlepas.
Kelas SCD berisi EKG sungguhan. **Tugas yang dikerjakan tesis adalah "ada EKG lawan tidak ada
EKG", bukan "SCD lawan normal".**

Satu angka membuktikannya. Regresi logistik atas **simpangan baku sinyal saja**:

| Himpunan | AUC | Akurasi |
|---|---|---|
| E0, jendela tesis | **1,000** | **1,000** |
| E1, jendela diperbaiki | 0,533 | 0,500 |
| E2, awitan digeser 60 menit | 0,605 | 0,541 |

(E2 tanpa rekaman 52: jendelanya memuat cuplikan kosong.)

### 1.1 Berlaku untuk semua durasi tesis, bukan hanya 3 menit

Tesis Tabel 4.4 melaporkan jendela 30 detik, 1, 2, 3, 4, 5, dan 10 menit. Notebook yang tersisa
untuk durasi itu: `Preprocessing-new.ipynb` (30 detik), `Preprocessing-new1.ipynb` (3 menit),
`Preprocessing.ipynb` (versi awal), `Timeseries - Preprocessing.ipynb` (blok 1 menit). **Semuanya
mengambil jendela Normal dari akhir rekaman `nsrdb`** (`start = time - m`); tidak satu pun
mengambil acak seperti tertulis di tesis bab 4.2. Notebook untuk 1, 2, 4, 5, dan 10 menit tidak
ada, tetapi polanya sama di semua yang tersisa: `m` = milidetik / 4 cuplikan dari akhir.
Jendela terpanjang, 10 menit, = 150.000 cuplikan = 19,5 menit pada 128 Hz — masih di dalam ekor
tanpa EKG yang paling pendek (63,8 menit). Jadi setiap jendela Normal tesis, pada semua durasi,
jatuh di bagian rekaman sesudah elektroda terlepas.

## 2. Temuan data lain

- **Panjang jendela tidak sama antarkelas.** `m = 45.000` cuplikan = 180 detik pada 250 Hz
  (`sddb`), tetapi **351,6 detik** pada 128 Hz (`nsrdb`).
- **`sddb` direkam 250 Hz**, bukan 256 Hz seperti tertulis di tesis bab 3.3.
- **`readlines()[:-1]` membuang rekaman 19830**; data sebenarnya 20 SCD + 18 Normal = 38.
- **6 dari 20 jendela SCD memuat label fibrilasi atrium** (rekaman 33, 36, 38, 44, 47, 52,
  anotasi `.ari`). Kelas SCD tidak seragam irama sinus.
- `LSTM` di `Modeling-new.ipynb` berjalan sepanjang **tinggi** citra (amplitudo), bukan lebar
  (waktu), karena `ReshapeLayer` memakai `shape[1]`.

## 3. Hasil model

| Himpunan | Model | Pengaturan | AUC [95 %] | Akurasi | Sens. | Spes. |
|---|---|---|---|---|---|---|
| E0 tesis | 2D-CNN | benar | 0,90 [0,74, 1,00] | 0,947 | 1,00 | 0,889 |
| E0 tesis | CNN-LSTM (sumbu tesis) | tesis: lr 1e-6, softmax ganda | 0,49 [0,30, 0,68] | 0,474 | 0,50 | 0,444 |
| E0 tesis | CNN-LSTM (sumbu waktu) | benar | 0,44 [0,24, 0,62] | 0,526 | 1,00 | 0,000 |
| E0 tesis | CNN-LSTM (sumbu waktu) | benar, citra dibalik | 0,79 [0,62, 0,93] | 0,658 | 1,00 | 0,278 |
| E1 diperbaiki | 2D-CNN | benar | **0,93 [0,80, 1,00]** | 0,895 | 0,90 | 0,889 |
| E1 diperbaiki | CNN-LSTM (sumbu waktu) | benar | 0,43 [0,24, 0,63] | 0,526 | 1,00 | 0,000 |
| E1 diperbaiki | CNN-LSTM (sumbu waktu) | benar, citra dibalik | **0,90 [0,77, 1,00]** | 0,895 | 0,85 | 0,944 |
| E2 awitan −60 menit | 2D-CNN | benar | **0,90 [0,76, 1,00]** | 0,895 | 0,90 | 0,889 |

Akurasi, sensitivitas, spesifisitas pada ambang 0,5 atas peluang rata-rata per pasien. Nilai per
ulangan ada di `keluaran/hasil_*.csv`; misalnya E1 2D-CNN 0,919 / 0,925 / 0,922.

### 3.1 Angka 96,67 % bisa didekati — dan tidak berarti apa-apa

2D-CNN yang diperbaiki pada jendela tesis mencapai akurasi 0,947, dekat 96,67 % tesis. Tetapi
§1 menunjukkan simpangan baku sinyal saja sudah memberi 1,000. Angka tesis mengukur apakah
citra berisi detak jantung.

### 3.2 LSTM gagal karena cara melatih, bukan karena data

Dengan pengaturan tesis, CNN-LSTM di sekitar tebakan acak (AUC 0,49). Memperbaiki laju belajar,
galat, dan sumbu waktu **belum cukup**: model tetap menebak satu kelas (sensitivitas 1,00,
spesifisitas 0,00, galat terpaku 0,69 ≈ ln 2). Pemeriksaan diagnostik (latih pada ke-38 citra
E1, tanpa uji) menunjukkan model bahkan tidak bisa mencocokkan data latihnya: akurasi latih
0,526 pada lr 1e-4 maupun 1e-3 sampai epoch 30.

Penyebabnya masukan: citra hampir seluruhnya latar putih (nilai 1), jadi LSTM menerima ribuan
masukan jenuh. Dengan citra dibalik (latar 0, jejak 1) akurasi latih naik ke 0,921 pada epoch
30, dan pada uji tingkat pasien CNN-LSTM mencapai AUC 0,90 — setara 2D-CNN. Tesis memakai latar
putih yang sama.

**Kesimpulan tesis bab 4.4 bahwa data "tidak cocok dengan karakteristik model LSTM" tidak
didukung.** LSTM belajar setelah masukannya diperbaiki. Pada E0 citra dibalik, model belajar
sebagian (AUC 0,79; satu lipatan terakhir galat 0,686) — 30 epoch batas bawah.

⚠️ Pembalikan citra ditemukan **sesudah** melihat LSTM gagal, jadi itu satu penyesuaian yang
tidak direncanakan sebelumnya. Dilaporkan terpisah dengan label "citra dibalik", tidak
menggantikan baris aslinya.

### 3.3 Data diperbaiki: akurasi tetap tinggi, maknanya tidak

Pada E1 — irama sinus sungguhan dari bagian `nsrdb` yang teranotasi, 3 menit, diubah ke 250 Hz
— 2D-CNN tetap mencapai AUC 0,93. Dua uji menunjukkan yang dipelajari bukan kedekatan dengan
kematian:

**Awitan digeser.** Jendela SCD yang berakhir **60 menit sebelum VF** (E2) memberi AUC 0,90,
praktis sama dengan jendela tepat sebelum VF (0,93). Model yang benar-benar melihat henti jantung
yang mendekat seharusnya melemah menjauhi awitan.

**Kontrol sakit tetapi hidup.** Model E1 menilai 15 pasien `chfdb` (gagal jantung berat, tidak
meninggal selama rekaman):

| Model | Pasien CHF dinyatakan SCD | Peluang SCD median: CHF / Normal / SCD |
|---|---|---|
| 2D-CNN | **86,7 %** (13 dari 15) | 0,90 / 0,04 / 0,90 |
| CNN-LSTM, citra dibalik | **86,7 %** | 0,79 / 0,21 / 0,75 |

Model memisahkan **pasien jantung dari relawan sehat**, bukan orang yang akan mengalami SCD dari
yang tidak. Ini temuan yang sama dengan `../Uji_Horizon/Hasil_Uji_1.md` (ciri HRV, regresi
logistik dan random forest), sekarang pada model citra tesis sendiri.

---

## 4. Jawaban atas pertanyaan

| Klaim tesis | Setelah diperbaiki |
|---|---|
| 2D-CNN mendiagnosa SCD dengan akurasi 96,67 % | Angka dapat didekati (0,947), tetapi pada data tesis tugasnya "ada EKG lawan tidak ada EKG"; simpangan baku saja memberi 1,000 |
| LSTM tidak cocok dengan data | Tidak didukung. LSTM gagal karena masukan latar putih; setelah dibalik, AUC 0,90 pada data diperbaiki |
| Model dapat mengklasifikasi beberapa menit sebelum SCD | Tidak didukung. Pada data diperbaiki AUC sama 60 menit sebelum VF (0,90) dan 87 % pasien gagal jantung yang hidup dinyatakan SCD |

**Yang boleh dikatakan:** model citra 2D (CNN atau CNN-LSTM) memisahkan rekaman pasien `sddb` dari
relawan sehat `nsrdb` dengan AUC sekitar 0,9 pada 38 pasien. Itu bukan diagnosis atau prediksi
SCD.

## 5. Batas

- 38 pasien; selang AUC lebar ([0,77, 1,00] dan sejenisnya). Yang kuat adalah **arah** temuan
  (tidak turun menjauhi VF, CHF dinyatakan SCD), bukan nilai titiknya.
- PyTorch, bukan Keras; LSTM tanh, bukan relu; citra digambar pada dpi 40 lalu diperkecil, bukan
  dpi 400. Model setara secara arsitektur, bukan salinan bobot.
- Pengaturan "tesis" dilatih 30 epoch, bukan 200.
- Jendela Normal E1 diambil di tengah rentang teranotasi, satu per rekaman; jam pengambilan tidak
  dipadankan dengan jendela SCD.
- Pembalikan citra adalah penyesuaian sesudah melihat hasil (§3.2).
