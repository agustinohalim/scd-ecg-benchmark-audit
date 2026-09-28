# SCD ECG benchmark audit

A master's thesis on ECG-based sudden cardiac death (SCD) classification, re-examined by its
author.

**KLASIFIKASI SINYAL ELEKTROKARDIOGRAM UNTUK MENDIAGNOSA SUDDEN CARDIAC DEATH MENGGUNAKAN 2D CNN DAN LSTM**
*(Electrocardiogram signal classification for diagnosing sudden cardiac death using 2D CNN and LSTM)*

Master's thesis, Magister Teknik Informatika, Universitas Bina Nusantara, 2022.
Author: Agustino Halim. Supervisor: Dr. Sani Muhamad Isa.

This repository holds the original 2022 code and a 2026 reanalysis by the author that fixes the
program, checks the data, and re-tests the thesis claims.

> **Read this first.** The reanalysis shows that the 2022 results do not measure what the thesis
> says they measure. The "Normal" class in the thesis contains no heartbeats (see below). Do not
> cite the 2022 accuracy (96.67 %) as evidence that ECG images can diagnose or predict sudden
> cardiac death.

## Repository layout

| Folder | Content |
|---|---|
| `original_2022/` | The thesis notebooks as they were run on Google Colab (preprocessing, 2D-CNN, CNN-LSTM, time-series variants), and `VF Data.csv` (VF onset per `sddb` record) |
| `reanalysis_2026/` | Scripts `00`–`05`, results in `results/`, and the full write-up `Hasil_Uji_Tesis.md` (Indonesian) |

## What the 2022 thesis did

- SCD class: a window of `sddb` channel 0 ending at ventricular fibrillation (VF) onset, 20
  records. Window lengths 30 s, 1, 2, 3, 4, 5 and 10 min (thesis Table 4.4), one model per length.
  Time-series variants used 2–10 consecutive images per record.
- Normal class: the same number of samples (`m` = milliseconds / 4) counted back from the **end**
  of each `nsrdb` record. The thesis text (section 4.2) says the Normal window was taken at random;
  none of the surviving notebooks does that.
- Each window drawn as a line plot, converted to a grayscale image, classified by a 2D-CNN or a
  2D-CNN + LSTM.
- Reported: 2D-CNN accuracy 96.67 % (1–3 min windows); CNN-LSTM "not suited to the data".

Notebooks survive for the 30 s and 3 min windows and the 1-minute time-series blocks; those for
1, 2, 4, 5 and 10 min do not. The reanalysis re-runs the 3 min window.

## What the 2026 reanalysis found

### The data

| Check | Result |
|---|---|
| Normal windows with any annotated heartbeat | **0 of 18** |
| Minutes between the last beat annotation and the end of each `nsrdb` record | 63.8 – 297.0 |
| Signal SD, thesis Normal window vs. mid-recording (median) | 0.065 mV vs. 0.29 mV |
| Classifier using only the signal's standard deviation, thesis windows | **AUC 1.000, accuracy 1.000** |

The last 45,000 samples of every `nsrdb` record come from after the electrodes were removed:
flat, quantised noise of about ±0.05 mV with no QRS complexes (`results/cek_jendela.png`,
`results/cek_ekor_nsrdb.png`). The thesis task was therefore "ECG vs. no ECG".

This holds for every window length in the thesis, not only 3 minutes. The longest, 10 min, is
150,000 samples, which at 128 Hz is 19.5 min of recording; the shortest no-ECG tail is 63.8 min.
Every thesis Normal window, at every length, lies inside it.

Other issues found in the code and text:

- 45,000 samples is 180 s at 250 Hz (`sddb`) but 351.6 s at 128 Hz (`nsrdb`): the two classes had
  different window lengths.
- `sddb` is recorded at 250 Hz, not 256 Hz as stated in the thesis.
- `readlines()[:-1]` silently dropped `nsrdb` record 19830.
- The LSTM ran along image height (amplitude), not width (time).
- Training used Adam lr 1e-6, a softmax layer followed by a loss expecting logits, the test set as
  validation set, three identical colour channels, and an augmentation generator that was never
  used.
- 6 of 20 SCD windows carry atrial fibrillation rhythm labels.

### The models, fixed

2D-CNN as in thesis Table 3.2 and a CNN-LSTM, rewritten in PyTorch; Adam 1e-4, cross-entropy on
logits, 30 fixed epochs, no test data seen during training. One image per record, patient-level
stratified 5-fold cross-validation repeated 3 times; AUC with 95 % patient-bootstrap intervals.

| Data | Model | AUC [95 % CI] | Accuracy |
|---|---|---|---|
| Thesis windows | 2D-CNN | 0.90 [0.74, 1.00] | 0.947 |
| Thesis windows | CNN-LSTM, thesis settings | 0.49 [0.30, 0.68] | 0.474 |
| Corrected: annotated sinus rhythm, 3 min, 250 Hz | 2D-CNN | **0.93 [0.80, 1.00]** | 0.895 |
| Corrected | CNN-LSTM, inverted images | **0.90 [0.77, 1.00]** | 0.895 |
| Corrected, SCD window ending **60 min before VF** | 2D-CNN | **0.90 [0.76, 1.00]** | 0.895 |

Living patients with severe heart failure (`chfdb`, 15 patients, never used in training) scored by
the corrected models: **13 of 15 classified as SCD** by both the 2D-CNN and the CNN-LSTM.

**Reading.** After the data is corrected the models still separate the two databases well, but
accuracy does not fall when the window moves an hour away from VF, and living heart-failure
patients are called SCD. The models separate cardiac patients from healthy volunteers; they do not
detect approaching sudden cardiac death. This matches a separate HRV-based analysis of the same
benchmark design.

**The LSTM.** With the thesis settings, and even with learning rate, loss and time axis fixed, the
CNN-LSTM could not fit its own training data (training accuracy stuck at the class share). The
images are almost entirely white background, which saturates the LSTM input. Inverting the images
(background 0, trace 1) lets it learn (AUC 0.90). The thesis conclusion that the data does not suit
an LSTM is not supported. The inversion was found after seeing the failure and is reported as a
separate row.

### Multi-window retest

`05_banyak_jendela.py` uses six 3-minute windows per patient instead of one (SCD windows ending
0–50 min before VF; Normal and CHF windows spread over the annotated recording), still split by
patient. Results: see `Hasil_Uji_Tesis.md` §6 once the run completes.

## Reproduce

Python 3.12 with `wfdb`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `Pillow`,
`torch` (CPU is enough; about 35 minutes per configuration).

```bash
cd reanalysis_2026
python 00_unduh_data.py            # PhysioNet sddb, nsrdb (about 1.8 GB), chfdb annotations
python 01_periksa_data.py          # data checks -> keluaran/periksa_data.csv
python 02_citra.py                 # images for E0 (thesis), E1 (corrected), E2 (-60 min), CHF
python 04_garis_dasar.py           # one-feature baseline
python 03_latih.py E1 cnn benar --chf
python 03_latih.py E1 lstm benar --chf --balik
python 05_banyak_jendela.py        # six windows per patient
```

Set `THESIS_DATA` to use data stored elsewhere. Scripts write to `keluaran/`; the committed
outputs are in `results/`. Code comments and identifiers are in Indonesian.

Differences from the 2022 setup: PyTorch instead of Keras; the LSTM uses tanh (PyTorch has no
ReLU LSTM); images drawn at 40 dpi and resized to 800 × 400 rather than 400 dpi; the "thesis
settings" run used 30 epochs, not 200. With 38 patients, intervals are wide: the direction of the
findings is solid, the point values are not.

## Data and licenses

Data are not included. MIT-BIH Sudden Cardiac Death Holter Database (`sddb`), MIT-BIH Normal
Sinus Rhythm Database (`nsrdb`) and BIDMC Congestive Heart Failure Database (`chfdb`) are
distributed by [PhysioNet](https://physionet.org) under the Open Data Commons Attribution
License v1.0. Cite each database as its PhysioNet page instructs, and PhysioNet itself:

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., Mietus, J. E.,
Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
Components of a new research resource for complex physiologic signals. *Circulation*, 101(23),
e215–e220.

Code and text in this repository are released under the MIT License (`LICENSE`).

## Citation

Halim, A. (2022). *Klasifikasi sinyal elektrokardiogram untuk mendiagnosa sudden cardiac death
menggunakan 2D CNN dan LSTM* [Master's thesis, Universitas Bina Nusantara].

The thesis results were also published as Halim, A., & Isa, S. M. (2023). Electrocardiogram
signal classification for diagnosis sudden cardiac death using 2D CNN and LSTM. *International
Journal of Intelligent Systems and Applications in Engineering*, 11(4s), 558–564. The reanalysis
above applies to those results as well.
