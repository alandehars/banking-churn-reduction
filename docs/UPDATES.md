# Catatan pembaruan

## 26 Oktober 2026 - V8 audit dan rencana pilot

Notebook V8 baru menambahkan audit model beku dengan pemeriksaan overlap terhadap seluruh CSV
development, kematangan/definisi label, interval metrik, serta simulasi kebutuhan sampel pilot
retensi. Form, penjelasan metode, batas interpretasi dan rekomendasi berbahasa Indonesia tersedia
di notebook. Panduan: [V8_GUIDE.md](V8_GUIDE.md).

V6/V7 dan hasil Colab V7 tetap dipertahankan. Paket core/model tidak dilatih ulang dan tidak ada
klaim peningkatan akurasi, validasi eksternal atau dampak kampanye baru. Default V8 adalah demo
sintetis. Audit V8 tidak mewarisi klaim eksternal dari pilihan mode helper batch lama.

CI diperluas untuk menguji sumber dan menjalankan seluruh delapan sel V8 pada empat kombinasi
Windows/Linux × Python 3.12/3.13, di samping seluruh pemeriksaan V7 yang dipertahankan.

## 26 Oktober 2026 - verifikasi penyimpanan V7

Hasil run Colab 24 Oktober sudah tersimpan lengkap dalam notebook lokal di folder project.
Verifikasi memeriksa 14 sel kode yang dieksekusi, nol error, 13 pemeriksaan internal, hash CSV
referensi, dan kesamaan kode dengan sumber V7 public. Identitas run serta metrik aktual
tercantum pada [verification.json](verification.json).

Skrip persiapan rilis lokal diperbaiki agar salinan sumber kosong masuk ke `source/v7/`.
Notebook lokal yang memuat hasil eksekusi dipertahankan; persiapan rilis tidak lagi menimpanya.
README folder project kini menunjuk V7, sementara petunjuk V6/V5 tetap tersedia sebagai arsip.

Pembaruan ini memperbaiki penyimpanan dan keterlacakan hasil. Tidak ada training ulang pada CSV referensi,
perubahan model, penggantian dataset, atau klaim peningkatan performa. Nilai simulasi tetap
bersyarat pada asumsi, dan test referensi tetap audit internal yang telah dilihat saat pengembangan.
