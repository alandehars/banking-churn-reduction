# Bukti validasi V7 dan arsip V6

## V7 - 24 Oktober 2026

- Environment Windows Python 3.12.14 dipasang dari `uv.lock` dengan uv 0.13.5; 16 dependensi akar dipatok, dependensi transitif serta marker OS dicatat.
- **48 pengujian lulus**: 30 kasus keputusan V6 dipertahankan, 18 kasus V7 mencakup training sintetis lengkap, nested CV, kalibrasi, threshold validation, jalur threshold tetap tanpa kalibrasi, reload bundle/helper, scoring CSV, serta laporan/ZIP.
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
