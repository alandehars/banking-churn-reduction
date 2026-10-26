# V8 — audit model beku dan perencanaan pilot

V7 tetap menjadi alur training. V8 adalah notebook terpisah dengan delapan sel kode dan delapan
bagian penjelasan. Ia menggunakan bundle dan kebijakan yang sudah dikunci, tanpa training ulang.
Paket utama tetap 7.0.0 karena tidak ada perubahan pada model/core V7; modul `v8_*` adalah suplemen.
Notebook V6/V7 beserta hasil referensi yang telah disimpan dipertahankan.

## Jalankan

Klik [Colab V8](https://colab.research.google.com/github/alandehars/banking-churn-reduction/blob/main/notebooks/banking_churn_v8.ipynb)
lalu **Runtime > Run all**. Default `demo_sintetis` membangkitkan CSV fiktif dan memakai aturan
skor deterministik yang tidak dilatih. Hasilnya hanya demonstrasi software.

Untuk bundle V7 milik Anda yang tepercaya:

1. Unggah `bank_churn_bundle.joblib` dan CSV development yang sesuai ke sesi pribadi Anda.
2. Pilih `bundle_tepercaya`, isi `Path_Bundle` serta `Path_Referensi`.
3. Isi `Path_Batch_Baru` bila sudah memiliki batch baru. Kosong berarti demo sintetis dengan model
   Anda; ini tidak menjadi validasi performa baru.
4. Lengkapi sumber, populasi, periode, horizon serta deklarasi kematangan label dan independensi.
5. Run all, baca rekomendasi dan interval, lalu unduh HTML/JSON hasil dari folder run baru.

Bundle joblib hanya boleh dimuat dari sumber tepercaya dengan versi training yang cocok.
V8 mempertahankan file bundle dan hash kebijakannya; tidak menyalin bundle ke repo.

Lokal, gunakan environment terkunci yang sama:

```sh
uv run --locked python scripts/run_notebook_v8.py --smoke
uv run --locked python scripts/run_notebook_v8.py --bundle /trusted/run/bank_churn_bundle.joblib --reference /private/Churn_Modelling.csv --output outputs/v8-frozen-demo.ipynb
```

Perintah kedua memakai batch sintetis karena `--batch` tidak diisi. Gunakan path output baru;
runner menolak menimpa file notebook yang sudah ada. Untuk batch nyata tambahkan `--batch`, lalu
lengkapi deklarasi sumber/label langsung di notebook sebelum membaca metrik prediksi.

## Perubahan substantif

- Pemeriksaan hash file, overlap ID dan fingerprint baris fitur memakai seluruh data development.
  Subset atau urutan baris baru tidak otomatis menjadi data independen. ID dinormalisasi setelah
  dibaca sebagai teks, sehingga nol di depan terjaga dan ID kosong/duplikat ditolak.
- Label belum matang atau tidak tersedia menunda metrik prediksi, sambil tetap menyediakan skor.
  Target tidak valid memblokir batch tanpa skor parsial. Satu kelas membuat AUC/AP tidak terdefinisi;
  denominator recall/precision nol dilaporkan sebagai kosong, bukan nol.
- Precision/recall dan proporsi memakai Wilson 95%; AUC/AP/Brier/log loss memakai paired row bootstrap.
  Interval mengasumsikan baris independen dan tidak mencakup seluruh ketidakpastian pengembangan model.
- Rencana pilot memakai dua kelompok seimbang, target penurunan absolut, alpha/power dan inflasi
  kehilangan outcome. Simulasi alternatif dan nol efek memeriksa deteksi serta false positive secara
  Monte Carlo. Efek dan risiko kontrol adalah asumsi, bukan estimasi dampak kampanye.

Batch helper V6/V7 lama menyebut `is_external_validation` berdasarkan nama mode `external_csv`.
Gunakan audit V8 untuk penilaian kesiapan: pemilihan mode saja tidak membuktikan independensi.
Kode lama dipertahankan sebagai arsip yang dapat direproduksi.

## Arti status

| Status | Implikasi |
|---|---|
| `blocked_invalid_schema` | Perbaiki input; tidak ada skor parsial. |
| `synthetic_demonstration` | Uji software saja, termasuk bila memakai model V7 asli. |
| `review_required` | Ada bukti yang belum lengkap, overlap, atau alarm kualitas/sampel. |
| `ready_for_independent_review` | Pemeriksaan yang tersedia telah dilewati; masih perlu review manusia. |

Tidak ada status deployment otomatis. `external_validation_verified` selalu `false` karena sumber,
periode dan horizon adalah deklarasi yang belum disahkan software. Pencocokan tidak menemukan
pelanggan yang ID dan fiturnya sama-sama berubah; nol overlap tidak membuktikan independensi.
Hash ID/fingerprint referensi hanya dipakai dalam memori lokal, tidak masuk ringkasan agregat.

## Output dan tindak lanjut

Run V8 menyimpan HTML mandiri, JSON audit, tabel metrik/kualitas/kalibrasi, rencana pilot, ringkasan
simulasi dan rekomendasi. `scores.csv` berisi skor per pelanggan dan harus tetap private. Metadata
sumber yang diketik pengguna juga perlu ditinjau sebelum ringkasan dibagikan. Data/model/hasil
pelanggan tidak masuk allowlist publikasi.

Tindak lanjut substantif: konfirmasi target dan horizon, kumpulkan populasi/periode baru dengan
label matang, audit model beku, kemudian jalankan pilot acak dengan kontrol. Menaikkan nomor versi,
mengganti seed, dan menjalankan demo tidak memberikan bukti performa atau dampak bisnis baru.

Referensi: [Wilson, NIST](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[normal power, statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.solve_power.html),
[Cohen h untuk dua proporsi](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_effectsize.html).
