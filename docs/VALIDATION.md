# Bukti validasi V6

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
