# Model card ringkas - V7

**Suplemen V8:** notebook audit dan perencanaan pilot memakai model/kebijakan V7 yang beku.
Ia tidak melatih model baru. Pemeriksaan overlap, label matang dan interval mengidentifikasi
celah bukti; `external_validation_verified` selalu false. Metadata sumber tetap deklarasi pengguna.
Demo sintetis dan simulasi efek pilot tidak menjadi hasil validasi/retensi nyata. Lihat
[V8_GUIDE](V8_GUIDE.md). Core paket serta prosedur training V7 di bawah tetap berlaku.

V7 menyatukan training, inferensi, simulasi, dan ekspor dalam paket Python 7.0.0. Notebook mandiri
menyematkan sumber modul yang sama dengan hash; CLI dan pengujian mengimpor paket tersebut.
Dependensi dikunci untuk Python 3.12/3.13. Metrik data referensi lokal sama dengan V6; perubahan ini
meningkatkan reproduksi dan keterujian, bukan bukti performa lebih baik.

Pada Colab, `requirements-colab-lock.txt` menyertakan closure dependensi analisis dari `uv.lock`
dengan hash yang sama dan mempertahankan kernel/UI yang dikelola layanan. Paket kernel tidak
dipasang ulang oleh notebook. Versi aktual analisis maupun paket host dicatat dalam metadata run.

Mode `synthetic_smoke` sengaja memakai data fiktif dan pencarian kecil. Konteks tersebut terpisah
dari `reference_source` dan `external_reviewed`, serta ditandai pada setiap ringkasan. Snapshot
`docs/reports/` adalah agregat dari satu run referensi lokal yang identitas lengkapnya ada pada
`public_summary.json`; snapshot repo tidak berubah mengikuti form pada salinan Colab pengguna.

Tujuan: studi kasus pembelajaran/portofolio dan eksplorasi kebijakan prioritas retensi.
Target `Exited` mengikuti file sumber; horizon waktunya belum terverifikasi.

Dataset referensi: 10.000 baris, 14 kolom, 2.037 label churn. `RowNumber`, `CustomerId`, dan `Surname`
tidak dipakai sebagai fitur. Sepuluh fitur model tercantum pada konfigurasi notebook. Sumber tidak
memastikan tanggal pengamatan, penerbit awal, mata uang, atau lisensi redistribusi; CSV tidak dikemas.

Metode: split terstratifikasi 60/20/20, nested CV 3×3 pada train, enam konfigurasi per pencarian,
seleksi menurut average precision, kalibrasi pada train, dan kebijakan threshold dari validation.
Kandidat mencakup Dummy, Logistic Regression, Random Forest, HistGradientBoosting, serta XGBoost.

Test merupakan holdout internal yang telah dipakai secara iteratif. Pemilihan kebijakan setelah
melihat test akan menambah bias; keputusan berikutnya perlu validation/data baru sesuai prosedur.
AP, precision, recall, kalibrasi, dan jumlah kontak menjawab pertanyaan berbeda. Nilai biaya serta
keberhasilan retensi tidak diestimasi dari label churn saja.

Form bisnis tidak mengganti kebijakan final yang diaudit. Estimasi nilai memakai asumsi risiko
tanpa kampanye, keberhasilan seragam, manfaat seragam, dan independensi bersyarat. Rentang parameter
simulator merupakan asumsi pengguna; ketidakpastian estimasi model/populasi belum tercakup.

Audit segmen menampilkan ukuran sampel dan error pada Age, Gender, dan Geography. Tinjau kelayakan
fitur serta dampak operasional dalam konteks penggunaan. Perbedaan error belum menjadi kesimpulan
fairness atau bukti kausal. Rekomendasi lanjutan: definisikan horizon, validasi data baru, konfirmasi
biaya/manfaat/kapasitas, lalu jalankan pilot terkontrol dengan ukuran sampel yang direncanakan.

Rujukan metode: [threshold](https://scikit-learn.org/stable/modules/classification_threshold.html),
[kalibrasi](https://scikit-learn.org/stable/modules/calibration.html),
[nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
[eksperimen terkontrol](https://www.microsoft.com/en-us/research/publication/online-experimentation-at-microsoft/).
