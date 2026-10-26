# Bukti validasi V8, V7 dan arsip V6

## Suplemen V8 - 26 Oktober 2026

V8 telah dijalankan pada kernel Python lokal dalam dua mode: aturan skor/data fiktif sepenuhnya,
dan bundle V7 tepercaya `20261024T132929Z-2d83fbd6` dengan 1.000 catatan sintetis. Masing-masing
mengeksekusi delapan sel tanpa error. Model serta kebijakan tetap beku. Ini tidak menyatakan
V8 telah dijalankan pada Google Colab atau tersedia data validasi bank yang baru.

**68 pengujian lokal lulus**, mencakup 49 kasus lama dan 19 kasus V8. Pengujian baru memeriksa
overlap subset/ID yang diganti, schema/ID tidak valid, label belum matang, metrik satu kelas,
Wilson/denominator nol, bootstrap deterministik, kebutuhan sampel, power simulasi dan false positive.
Simulasi subset CSV development berhasil menemukan 50 ID dan 50 baris fitur yang sama meskipun
hash file subset berbeda. Pencocokan tidak membuktikan independensi bila nol overlap ditemukan.

Notebook hasil V7 dipertahankan dengan SHA-256 yang sama. Bukti V8, hash notebook baru, asumsi
pilot dan hasil Monte Carlo tersedia di [v8_verification.json](v8_verification.json). Rencana contoh
memerlukan 2.626 outcome per kelompok dan 5.836 rekrutmen total setelah inflasi outcome hilang
10%; churn kontrol 20%, target penurunan 3 poin persentase, alpha dua sisi 5%, power 80% adalah
asumsi perencanaan. Ini bukan efek retensi yang diamati.

## Verifikasi penyimpanan - 26 Oktober 2026

Ekspor hasil Google Colab run `20261024T135218Z-b828e79e` sudah tersimpan sebagai notebook
lokal dalam folder project. Seluruh **14 sel kode memiliki output eksekusi, tanpa error**, dan
**13 pemeriksaan internal lulus**. AST kode cocok dengan sumber V7 public, serta SHA-256 CSV
cocok dengan snapshot referensi. Lingkungan run: Python 3.12.13, 34 dependensi analisis cocok
dengan lock khusus Colab. Tanggal 26 Oktober merupakan tanggal verifikasi file, bukan run baru.

Metrik run Colab sama dengan referensi lokal di bawah: ROC-AUC 0,8653021280 dan AP 0,7164542530.
File lengkap yang diverifikasi adalah ekspor lokal; ini tidak menyatakan autosave output akhir
Drive berhasil. [verification.json](verification.json) mencatat lokasi penyimpanan secara eksplisit
dan mempertahankan bukti V6 sebagai arsip.

[CI pada kode V7](https://github.com/alandehars/banking-churn-reduction/actions/runs/40099605097)
lulus pada empat kombinasi Windows/Linux dan Python 3.12/3.13. Tiap job menjalankan 49 pengujian
serta seluruh sel notebook sintetis. Pengujian sintetis membuktikan alur software, bukan validasi
eksternal model. Skrip rilis lokal juga diperiksa agar tidak menghapus output notebook yang tersimpan.

## V7 - 24 Oktober 2026

- Environment Windows Python 3.12.14 dipasang dari `uv.lock` dengan uv 0.13.5; 16 dependensi akar dipatok, dependensi transitif serta marker OS dicatat.
- **49 pengujian lulus**: 30 kasus keputusan V6 dipertahankan, 19 kasus V7 mencakup training sintetis lengkap, nested CV, kalibrasi, threshold validation, jalur threshold tetap tanpa kalibrasi, reload bundle/helper, scoring CSV, serta laporan/ZIP.
- Notebook sintetis final: run `20261024T132911Z-366faab1`, **14 sel kode dieksekusi**, tujuh form, **13 pemeriksaan internal PASS**, tanpa error.
- Notebook referensi menjalankan alur yang sama pada CSV pengguna. `docs/reports/public_summary.json` mencatat run sumber, hash data/split, versi aktual, konfigurasi, dan seluruh asumsi ringkasan yang dipublikasikan.
- Hasil referensi lokal: HistGradientBoosting; ROC-AUC **0,8653021280**, AP **0,7164542530**, threshold F1 **0,29505758**; 449 kontak, TP 267, FP 182, FN 140. Metrik sama dengan V6 lokal, bukan klaim peningkatan akurasi.
- Laporan PDF dua halaman dirender ke PNG dan diperiksa tampilannya; HTML mandiri, JSON hanya agregat, arsip lolos CRC. Sumber mentah, bundle, serta skor pelanggan tidak masuk publikasi.
- CI dikonfigurasi untuk instalasi terkunci, pengujian, dan eksekusi setiap sel notebook sintetis pada Windows/Linux. Status setiap run dapat dilihat di [GitHub Actions](https://github.com/alandehars/banking-churn-reduction/actions/workflows/ci.yml).
- [Colab V7](https://colab.research.google.com/drive/1OlYlT8VhD7e323tNt0A9vm-comOiEIVU) dibuat sebagai file baru dalam folder Drive yang sama. Bukti eksekusi cloud dilaporkan terpisah dari run lokal.

Pengujian sintetis memverifikasi alur software, bukan performa bank. Test sumber telah dipakai secara iteratif, tanpa timestamp maupun treatment/control. Validasi temporal/eksternal dan pilot terkontrol tetap pekerjaan lanjutan. Mengganti seed pada CSV sama tidak menyediakan validasi baru.

## Arsip bukti V6

Tanggal: 24 Oktober 2026. Kode public notebook sama dengan versi yang diuji; output dan metadata
identitas dibersihkan untuk distribusi. Repo ini tidak menyertakan artefak pelanggan atau bundle.

| Pemeriksaan pipeline | Bukti |
| --- | --- |
| Run lokal | `20261024T063308Z-a1691734` |
| Run Google Colab | `20261024T063629Z-b9a63013` |
| Eksekusi Colab | 38 sel kode, 102 pemeriksaan internal, 0 error |
| Baca ulang Drive | Struktur dan kode dibandingkan dengan versi lokal |
| Form kebijakan | Kapasitas nol menghasilkan tanpa kampanye dan biaya tetap nol |
| Form ketidakpastian | Kontak nol menghasilkan kuantil net nol |
| Form batch | Stres sintetis memunculkan alarm Age, Balance, dan Geography |
| Penyimpanan akhir | Nilai awal dipulihkan, laporan dan ZIP diperbarui |

Baseline V5 lokal dipertahankan dan metrik model V6 lokal sama pada lingkungan yang sama.
Pengembangan V6 memperluas alat keputusan serta pelaporan; bukan klaim kenaikan akurasi.

Pengujian pada repo public menggunakan 30 kasus dengan fixture sintetis dan kode validasi/inference
yang benar-benar tersemat pada notebook. Fixture tersebut tidak melatih model dan tidak menjadi
validasi eksternal atas dataset referensi. GitHub Actions memeriksa kode tersebut tanpa data sumber.
