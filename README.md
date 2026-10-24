# Banking Churn Reduction · V6

[![CI](https://github.com/alandehars/banking-churn-reduction/actions/workflows/ci.yml/badge.svg)](https://github.com/alandehars/banking-churn-reduction/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alandehars/banking-churn-reduction/blob/main/notebooks/banking_churn_v6.ipynb)

Studi kasus churn perbankan dengan penjelasan berbahasa Indonesia, evaluasi model yang memisahkan
train/validation/test, dan simulasi keputusan retensi. Notebook dapat dijalankan sendiri di Colab.
Versi public ini menyertakan kode dan dokumentasi; seluruh output serta metadata identitas notebook
dihapus sebelum publikasi. CSV mentah, model tersimpan, dan hasil scoring pelanggan tidak disertakan.

## Yang bisa dicoba

| Bagian | Isi |
| --- | --- |
| 2–5 | Verifikasi sumber, split 60/20/20, nested CV pada train, kalibrasi, dan kebijakan dari validation |
| 6–7 | Audit test, interpretasi otomatis, permutation importance, dan error per segmen |
| 10 | Empat form: profil, threshold, kapasitas, serta retensi/biaya |
| 12 | Rekomendasi, prioritas perbaikan, dan rancangan pilot treatment/control |
| 13.1 | Kandidat kebijakan menurut batas kontak, anggaran, dan asumsi nilai |
| 13.2 | Simulasi rentang asumsi biaya, manfaat, serta efektivitas |
| 13.3 | Perbandingan deskriptif dengan baseline V5 lokal |
| 13.4–13.5 | Kualitas CSV batch, scoring, laporan aktual, dan pemeriksaan |
| 14 | Ekspor laporan, model, tabel, snapshot form, dan ZIP setiap run |

## Mulai di Colab

1. Klik **Open in Colab**, lalu simpan salinan ke Drive Anda untuk menyimpan hasil.
2. Baca sumber serta konfigurasi, lalu pilih **Runtime → Run all**.
3. Jika CSV sumber tidak bisa diunduh, unggah file yang sesuai ke Files; lihat [catatan data](data/README.md).
4. Sesudah model siap, ubah parameter pada form lalu jalankan sel form terkait tanpa training ulang.
5. Unduh ZIP dari `Files → bank_churn_artifacts_v6`. Untuk demo batch, unggah
   [tiga profil sintetis](examples/synthetic_customers.csv), pilih **CSV saya**, dan isi path file.

Mode awal **Teks stabil** menyimpan grafik sebagai PNG dan mencetak hasil utama sebagai teks.
Form V6 memperbarui `current_report.md/json` dan ZIP run yang sama. Snapshot contoh bagian 10
tetap terpisah dari perubahan input form. [Colab V6 dengan hasil run pemilik proyek](https://colab.research.google.com/drive/1eFtkPTLDtvDf3ub860bznTk4FDAOlk_W)
memerlukan izin akses Drive sesuai pengaturan pemilik.

## Menjalankan lokal

Gunakan Python 3.12. Dari root repo:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_notebook.py
python -m unittest discover -s tests -v
python scripts/run_notebook.py
```

Letakkan CSV sumber di root repo bila tersedia. Output eksekusi lokal disimpan terpisah di
`outputs/banking_churn_v6.executed.ipynb`. Model dan tabel hasil dibuat di `bank_churn_artifacts_v6/`.
Direktori keluaran tersebut diabaikan Git. Jangan memasukkan hasil scoring pelanggan ke commit public.

## Kode dan pengujian

- `notebooks/banking_churn_v6.ipynb`: notebook mandiri dengan 38 sel kode dan tujuh form.
- `src/banking_churn/decision_tools.py`: fungsi keputusan, ketidakpastian, kualitas batch, dan laporan.
- `tests/`: 30 pengujian numerik, skema, dan integritas laporan/ZIP dengan fixture sintetis.
- `scripts/check_notebook.py`: validasi struktur, output kosong, dan kesamaan modul dengan kode tersemat.
- `scripts/sync_notebook.py`: sinkronkan perubahan modul ke notebook sebelum commit.

```sh
python scripts/sync_notebook.py
python scripts/check_notebook.py
python -m unittest discover -s tests -v
```

GitHub Actions menjalankan pemeriksaan notebook dan pengujian sintetis tanpa CSV sumber,
bundle privat, atau kredensial Drive. CI tidak melatih model atau membuktikan metrik pada data nyata.
Catatan eksekusi pipeline sebelumnya ada pada [laporan validasi](docs/VALIDATION.md).

## Interpretasi dan batas penggunaan

Pemilihan keluarga model dan parameter dilakukan pada train. Threshold dipilih pada validation.
Test lama tetap audit internal yang sudah dilihat selama pengembangan, sehingga validasi pada data
periode/populasi baru masih diperlukan. Baseline V5 menyimpan metrik agregat dan identitas pembagian
data; perubahan versi paket/hardware dapat memengaruhi perbandingan.

Skor churn dan permutation importance tidak mengukur penyebab churn atau dampak penawaran.
Biaya, manfaat, dan efektivitas pada simulator adalah asumsi; proporsi draw positif bukan peluang
keuntungan kampanye yang telah terverifikasi. Uji program retensi dengan kelompok kontrol sebelum
memperluas penerapan. Rincian ada di [MODEL_CARD](docs/MODEL_CARD.md).
