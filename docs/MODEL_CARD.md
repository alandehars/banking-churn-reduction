# Model card ringkas

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
