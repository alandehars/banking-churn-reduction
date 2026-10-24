"""Build the standalone V7 notebook from the same package source used in CLI and CI."""
import hashlib
import json
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def md(key,source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(),id=key))
    def code(key,source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(),id=key))
    md("intro",r"""
    # Banking Churn Reduction - V7

    Notebook analisis dan simulasi dengan penjelasan Indonesia. V7 memakai **modul Python yang sama** untuk Colab, CLI, dan uji otomatis. V6 tetap tersedia sebagai versi terdahulu.

    **Cara memakai:** jalankan dari atas ke bawah melalui **Runtime > Run all**. Form konfigurasi memilih CSV referensi atau demonstrasi sintetis. Setelah training selesai, ubah salah satu dari tujuh form simulasi lalu jalankan hanya sel tersebut. Laporan HTML/PDF dan ZIP diperbarui oleh form advisor, ketidakpastian, dan batch; empat form dasar menyimpan snapshot tersendiri.

    **Yang dipelajari:** memverifikasi data, memisahkan train/validation/test, memilih model tanpa memakai test, membaca metrik pada data tidak seimbang, dan menghubungkan prioritas churn dengan kapasitas serta asumsi ekonomi. Skor historis belum mengukur pelanggan yang terselamatkan.

    Seluruh output training dan scoring disimpan dalam folder run tersendiri. File public repo hanya berisi sumber dan ringkasan agregat. Hindari membagikan ZIP/model/hasil scoring bila memakai data pelanggan sebenarnya.

    [Repository dan dokumentasi](https://github.com/alandehars/banking-churn-reduction) | [Notebook V6](https://colab.research.google.com/github/alandehars/banking-churn-reduction/blob/main/notebooks/banking_churn_v6.ipynb)
    """)
    md("environment_notes","""
    ## 1. Lingkungan yang dapat direproduksi

    Python **3.12 dan 3.13** didukung; Colab terbaru memakai 3.13. Dependensi berikut berasal dari `uv.lock`; instalasi notebook memakai ekspor dengan hash, sedangkan CI memakai `uv sync --locked`. Windows dan Linux menjalankan versi akar yang sama, dengan dependensi sistem yang sesuai masing-masing OS/Python. Hasil numerik masih dapat sedikit berbeda menurut hardware.

    Sel ini memeriksa paket terpasang sebelum mengimpor modul analisis. Jika paket yang sudah dimuat perlu diganti, instalasi akan meminta **Runtime > Restart session**, lalu **Run all** lagi. Ini mencegah proses memakai campuran biner lama dan baru. Internet diperlukan hanya saat instalasi atau unduh CSV referensi; mode sintetis tidak mengunduh data.
    """)
    lock=(ROOT/"requirements-lock.txt").read_text(encoding="utf-8")
    deps='LOCKED_REQUIREMENTS = '+repr(lock)+'\n'+textwrap.dedent('''
        import importlib.metadata as package_metadata
        import os,subprocess,sys,tempfile
        from pathlib import Path
        from packaging.requirements import Requirement
        if sys.version_info[:2] not in {(3,12),(3,13)}:
            raise RuntimeError("V7 memerlukan Python 3.12 atau 3.13. Versi aktif: "+sys.version.split()[0])
        requirements=[]
        for line in LOCKED_REQUIREMENTS.splitlines():
            line=line.strip()
            if not line or line.startswith(("#","--")):continue
            entry=line.rstrip(chr(92)).strip()
            req=Requirement(entry)
            if req.marker is None or req.marker.evaluate():requirements.append(req)
        missing=[]
        for req in requirements:
            try:installed=package_metadata.version(req.name)
            except package_metadata.PackageNotFoundError:installed=None
            if installed is None or not req.specifier.contains(installed,prereleases=True):missing.append(req.name)
        if missing:
            if os.getenv("BANK_CHURN_SKIP_INSTALL")=="1":
                raise RuntimeError("Environment terkunci tidak cocok: "+", ".join(missing))
            aliases={"scikit-learn":"sklearn","pillow":"PIL","python-dateutil":"dateutil","pyyaml":"yaml"}
            loaded=any(aliases.get(name,name.replace("-","_")) in sys.modules for name in missing)
            lock_path=Path(tempfile.mkdtemp(prefix="bank-churn-lock-"))/"requirements-lock.txt"
            lock_path.write_text(LOCKED_REQUIREMENTS,encoding="utf-8")
            subprocess.check_call([sys.executable,"-m","pip","install","--quiet","--require-hashes","-r",str(lock_path)])
            if loaded:raise RuntimeError("Paket yang sudah dimuat telah diganti. Runtime > Restart session, lalu Run all kembali.")
        print("Python",sys.version.split()[0],"|",len(requirements),"distribusi terkunci cocok dengan environment ini.")
    ''').strip()
    code("dependencies",'#@title 1.1 - Verifikasi paket terkunci\n'+deps)
    modules={p.name:p.read_text(encoding="utf-8") for p in sorted((ROOT/"src/banking_churn").glob("*.py"))}
    hashes={name:hashlib.sha256(src.encode()).hexdigest() for name,src in modules.items()}
    bootstrap='PACKAGE_SOURCES = '+repr(modules)+'\nPACKAGE_SHA256 = '+repr(hashes)+'\n'+textwrap.dedent('''
        import hashlib,importlib,sys,tempfile
        from pathlib import Path
        package_root=Path(tempfile.mkdtemp(prefix="bank-churn-v7-"))
        package_dir=package_root/"banking_churn";package_dir.mkdir()
        for name,source in PACKAGE_SOURCES.items():
            assert hashlib.sha256(source.encode()).hexdigest()==PACKAGE_SHA256[name]
            (package_dir/name).write_text(source,encoding="utf-8")
        for name in list(sys.modules):
            if name=="banking_churn" or name.startswith("banking_churn."):del sys.modules[name]
        sys.path.insert(0,str(package_root));importlib.invalidate_caches()
        from banking_churn import __version__
        from banking_churn.config import TrainingConfig
        from banking_churn.pipeline import run_training,finish_run,initialize_decision_examples
        from banking_churn.inference import predict_customers,read_source_csv
        from banking_churn.decision_tools import v6_policy_advisor,v6_uncertain_campaign,v6_batch_quality,v6_json,v6_number
        from banking_churn.live import save_live_section,refresh_reports
        from banking_churn.simulation import profile_what_if,threshold_scenario,capacity_scenario,retention_scenario
        from banking_churn.synthetic import write_synthetic_csv
        import numpy as np,pandas as pd,json
        from IPython.display import display,Markdown,Image,FileLink
        print("Modul bersama V"+__version__,"|",len(PACKAGE_SOURCES),"berkas terverifikasi.")
    ''').strip()
    code("shared_package",'#@title 1.2 - Muat modul bersama yang terverifikasi\n'+bootstrap)
    md("source_notes","""
    ## 2. Sumber, cakupan, dan batas data

    CSV diberikan pengguna melalui [Drive sumber](https://drive.google.com/file/d/1_mpH_uzTGHaYIF33pz9Is6KW_wj8cv8-/view). Snapshot referensi memiliki **10.000 baris, 14 kolom**, dan **2.037 label churn**. SHA-256 referensi: `3996cd1fa372e0db0cd9c0ebac35bbd4e8e3c65fb942bb010c826e7b1eeef0a0`. Angka tersebut mendeskripsikan file ini; penerbit asal, waktu ekstraksi, horizon churn, mata uang, dan lisensi redistribusi belum terverifikasi.

    `Exited` adalah target 0/1. Enam fitur numerik, dua fitur biner, dan dua fitur kategori dipakai. `RowNumber`, `CustomerId`, serta `Surname` tidak menjadi prediktor. Pelanggan berulang dengan ID sama ditolak karena memerlukan split per pelanggan/waktu. Missing fitur ditangani dalam pipeline, tanpa mengisi target yang hilang. Nilai di luar domain dan kategori asing ditandai saat inferensi.

    Mode **synthetic_smoke** membangkitkan data fiktif berlabel untuk menguji semua tahap dengan waktu singkat. Ukuran data, distribusi, model terpilih, dan metriknya berbeda dari referensi. Semua laporan mode ini diberi label DATA SINTETIS. Keberhasilan CI sintetis membuktikan alur berfungsi, bukan akurasi pada bank nyata.
    """)
    code("configuration",'''#@title Konfigurasi V7
Mode_data = "reference_source" #@param ["reference_source", "synthetic_smoke"]
Path_CSV = "churn.csv" #@param {type:"string"}
Seed = 42 #@param {type:"integer"}
Jumlah_baris_sintetis = 1000 #@param {type:"integer"}
OUTPUT_BASE = Path("bank_churn_artifacts_v7")
DATA_FILE = Path(Path_CSV)
STRICT_REFERENCE = True
if Mode_data=="synthetic_smoke":
    import uuid
    DATA_FILE=OUTPUT_BASE/"inputs"/("synthetic-"+uuid.uuid4().hex[:8]+".csv")
    write_synthetic_csv(DATA_FILE,Jumlah_baris_sintetis,Seed)
    CONFIG=TrainingConfig.quick(Seed);STRICT_REFERENCE=False
elif Mode_data=="reference_source":
    CONFIG=TrainingConfig(seed=Seed)
    if not DATA_FILE.is_file():
        import gdown
        DATA_FILE.parent.mkdir(parents=True,exist_ok=True)
        downloaded=gdown.download(id="1_mpH_uzTGHaYIF33pz9Is6KW_wj8cv8-",output=str(DATA_FILE),quiet=False)
        if not downloaded:raise FileNotFoundError("Unduhan gagal. Unggah CSV sumber ke Files dan isi Path_CSV.")
else:raise ValueError("Mode data tidak dikenal")
CONFIG.validate()
print("Konteks:",Mode_data,"| sumber tersedia:",DATA_FILE.is_file(),"| seed:",Seed)
print("Data sintetis: uji alur saja." if CONFIG.smoke else "Hash CSV referensi akan diverifikasi sebelum training.")''')
    md("methods","""
    ## 3. Training, kalibrasi, dan pemilihan kebijakan

    Alur: **verifikasi CSV → split 60% train / 20% validation / 20% test → nested CV pada train → kalibrasi pada train → threshold pada validation → audit test → simulasi → ekspor**.

    Preprocessing terdiri dari imputasi numerik median + standardisasi, serta imputasi kategori + one-hot encoding. Semua transformer di-fit hanya di dalam fold training sehingga statistik validation/test tidak bocor. Model referensi mencakup Dummy, Logistic Regression, Random Forest, HistGradientBoosting, dan XGBoost. Outer CV 3 fold mengaudit pencarian parameter 3 fold; enam percobaan dibatasi untuk runtime. Pemilihan keluarga model memakai average precision dari train, dengan Dummy sebagai pembanding. Mode sintetis memakai tiga keluarga, 2×2 fold, dan satu percobaan.

    Kalibrasi sigmoid membungkus pencarian parameter sehingga setiap fold kalibrasi melatih ulang pencarian pada bagiannya sendiri. Skor validation dipakai untuk threshold F0,5, F1, dan F2. F0,5 memberi bobot precision lebih besar; F2 memberi bobot recall lebih besar; F1 menyeimbangkan keduanya. Default adalah F1. Ada pula threshold 0,5 serta kebijakan top-k berdasarkan kapasitas. Semua pilihan dikunci sebelum test dinilai. Jika nilai ranking sama, pemilihan top-k memakai urutan ID indeks yang stabil.

    Test ini sudah dilihat selama iterasi V5/V6/V7, sehingga tetap **audit internal**. Mengganti seed pada data yang sama tidak menghasilkan validasi eksternal. Dataset tidak memiliki timestamp yang memadai; temporal split yang sah memerlukan tanggal observasi dan horizon label baru.
    """)
    code("training",'''run=run_training(DATA_FILE,OUTPUT_BASE,CONFIG,Mode_data,STRICT_REFERENCE)
ARTIFACT_DIR=run.artifact_dir
print("Run:",run.run_id,"| model:",run.metadata["model_name"])
print(run.tables["split_summary"].to_string(index=False))
print("\\nNested CV pada train:")
print(run.tables["nested_model_comparison"].to_string(index=False))
print("\\nKalibrasi pada validation (diagnostik, bukan pemilihan test):")
print(run.tables["calibration_diagnostic"].to_string(index=False))''')
    md("metric_notes","""
    ## 4. Membaca audit model

    **ROC-AUC** mengukur ranking label positif terhadap negatif. **Average precision (AP)** merangkum precision-recall sepanjang threshold; bandingkan dengan prevalensi test, bukan dengan accuracy. **Brier/log loss** mengukur kesalahan probabilitas, lebih rendah lebih baik. **Precision** adalah bagian prioritas yang berlabel churn; **recall** adalah bagian seluruh churn yang tertangkap. TP/FP/FN/TN menjelaskan jumlah konkret, dan contact rate harus sesuai kapasitas tim.

    Interval bootstrap menggambarkan variasi resampling pada test tetap, bukan seluruh ketidakpastian proses training, perubahan populasi, atau dampak intervensi. Permutation importance dihitung pada validation, memperlihatkan ketergantungan prediksi terhadap fitur; fitur berkorelasi dapat saling berbagi informasi. Audit segmen diberi penanda sampel terbatas. Perbedaan segmen tidak langsung menjadi bukti diskriminasi atau penyebab churn.
    """)
    code("audit",'''print("Skor test:",json.dumps(v6_json(run.metadata["test_scores"]),indent=2))
print("\\nKebijakan dikunci pada validation; tabel berikut audit test:")
print(run.tables["test_policies"].to_string(index=False))
print("\\nInterval bootstrap test:",run.tables["test_bootstrap_intervals"].to_string(index=False),sep="\\n")
print("\\nPermutation importance validation:",run.tables["validation_importance"].to_string(index=False),sep="\\n")
print("\\nAudit segmen test:",run.tables["test_segments"].to_string(index=False),sep="\\n")
manifest=finish_run(run)
display(Image(filename=str(ARTIFACT_DIR/"model_audit.png")))
display(Markdown((ARTIFACT_DIR/"current_report.md").read_text(encoding="utf-8")))''')
    md("forms_guide","""
    ## 5. Laboratorium simulasi - tujuh form

    Semua form memakai estimator tetap; perubahan form tidak melatih model ulang. Form threshold, kapasitas, advisor, dan simulasi ekonomi memakai **validation**, agar eksplorasi tidak memilih kebijakan lewat test. Angka pada empat form dasar adalah snapshot eksplorasi, terpisah dari default audit model. Advisor, ketidakpastian, dan kualitas batch memperbarui ringkasan eksekutif run yang sama.

    **UM** adalah unit moneter asumsi. Rumus expected net adalah `B × u × jumlah risiko pelanggan terpilih - C × jumlah kontak - F`, dengan F hanya dibebankan bila ada kontak. Ini memerlukan asumsi bahwa skor mendekati risiko tanpa kampanye, u peluang menyelamatkan pelanggan yang akan churn, serta B manfaat kontribusi setelah biaya relevan. Dataset tidak mengestimasi u atau membuktikan asumsi tersebut. Zero kontak menghasilkan biaya dan net nol.
    """)
    code("form_state",'''X_train=run.frames["train"];X_valid=run.frames["validation"]
y_valid=run.labels["validation"];validation_probability=run.probabilities["validation"]
policies=run.metadata["policies"]
def save_exploration(kind,inputs,summary,tables=None):
    payload=v6_json({"kind":kind,"run_id":run.run_id,"source_sha256":run.metadata["data_sha256"],
        "data_context":Mode_data,"scope":"validation_exploration","inputs":inputs,"summary":summary})
    (ARTIFACT_DIR/("form_"+kind+"_latest.json")).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for key,table in (tables or {}).items():table.to_csv(ARTIFACT_DIR/("form_"+kind+"_"+key+"_latest.csv"),index=False)
    refresh_reports(ARTIFACT_DIR)
    print("Snapshot eksplorasi tersimpan. Default model tetap:",run.metadata["default_policy"])
print("Form siap | konteks:",Mode_data,"|",len(X_valid),"baris validation")''')
    md("profile_notes","""
    ### 5.1. What-if profil
    Bandingkan profil fiktif yang sama dengan status keaktifan atau jumlah produk berbeda. Selisih skor menunjukkan respons estimator, **bukan pengaruh kausal** perubahan produk/keaktifan. Periksa juga apakah kombinasi fitur masuk akal dan masih dalam dukungan training.
    """)
    code("form_profile",'''#@title Form 1 - What-if profil fiktif
Usia_profil = 45 #@param {type:"integer"}
Aktif_skenario = 1 #@param [0,1] {type:"raw"}
Produk_skenario = 2 #@param {type:"integer"}
baseline=pd.DataFrame([dict(CreditScore=650,Age=Usia_profil,Tenure=5,Balance=50000.,NumOfProducts=1,
    EstimatedSalary=60000.,HasCrCard=1,IsActiveMember=0,Geography="France",Gender="Male")])
profile_result=profile_what_if(baseline,{"IsActiveMember":Aktif_skenario,"NumOfProducts":Produk_skenario},run.bundle,run.metadata["threshold"],predict_customers)
print(profile_result["comparison"].to_string());print("Perubahan skor (poin persen):",profile_result["risk_change_pp"])
print("Bukan estimasi efek intervensi.")
save_exploration("profile",{"age":Usia_profil,"active":Aktif_skenario,"products":Produk_skenario},
    {"risk_change_pp":profile_result["risk_change_pp"]},{"comparison":profile_result["comparison"].reset_index()})''')
    md("threshold_notes","""
    ### 5.2. Threshold
    Threshold lebih rendah biasanya menambah prioritas dan recall, dengan tambahan false positive. Form ini hanya memperlihatkan tradeoff pada validation. Threshold model yang tersimpan tetap berasal dari konfigurasi training.
    """)
    code("form_threshold",'''#@title Form 2 - Eksplorasi threshold validation
Threshold_simulasi = 0.30 #@param {type:"number"}
threshold_result=threshold_scenario(y_valid,validation_probability,Threshold_simulasi)
print(json.dumps(v6_json(threshold_result["metrics"]),indent=2))
save_exploration("threshold",{"threshold":Threshold_simulasi},threshold_result["metrics"])''')
    md("capacity_notes","""
    ### 5.3. Kapasitas
    Top-k memilih risiko tertinggi sesuai proporsi batch, tanpa menjamin semua yang dipilih akan churn. Bandingkan recall dan precision pada kapasitas tim nyata; hasilnya bergantung pada komposisi batch. Zero persen berarti tidak ada kontak.
    """)
    code("form_capacity",'''#@title Form 3 - Kapasitas tim pada validation
Kapasitas_persen = 20.0 #@param {type:"number"}
fraction=v6_number(Kapasitas_persen,"Kapasitas",high=100)/100
capacity_result=capacity_scenario(y_valid,validation_probability,fraction,index=X_valid.index)
print(json.dumps(v6_json(capacity_result["metrics"]),indent=2))
save_exploration("capacity",{"capacity_percent":Kapasitas_persen},capacity_result["metrics"])''')
    md("retention_notes","""
    ### 5.4. Retensi dengan parameter tetap
    Monte Carlo membangkitkan jumlah retained fiktif menggunakan `risiko × u` untuk pelanggan terpilih. Kuantil net merangkum variasi outcome di bawah asumsi tetap; belum memasukkan kesalahan kalibrasi, heterogenitas treatment, atau perubahan populasi. Distribusi ini bukan laba yang telah diamati.
    """)
    code("form_retention",'''#@title Form 4 - Retensi dan biaya dengan asumsi tetap
Kontak_retensi_persen = 20.0 #@param {type:"number"}
Biaya_kontak_retensi_UM = 10.0 #@param {type:"number"}
Manfaat_retained_retensi_UM = 100.0 #@param {type:"number"}
Keberhasilan_retensi_persen = 20.0 #@param {type:"number"}
Biaya_tetap_retensi_UM = 500.0 #@param {type:"number"}
Jumlah_draw_retensi = 2000 #@param {type:"integer"}
retention_inputs=dict(fraction=v6_number(Kontak_retensi_persen,"Kapasitas",high=100)/100,
    contact_cost=Biaya_kontak_retensi_UM,value_per_saved=Manfaat_retained_retensi_UM,
    success_rate=v6_number(Keberhasilan_retensi_persen,"Keberhasilan",high=100)/100,
    fixed_cost=Biaya_tetap_retensi_UM,n_draws=Jumlah_draw_retensi,seed=Seed)
retention_result=retention_scenario(validation_probability,index=X_valid.index,**retention_inputs)
print(json.dumps(v6_json(retention_result["summary"]),indent=2))
print("Bersyarat pada asumsi, bukan dampak bisnis terukur.")
save_exploration("retention",retention_inputs,{"results":retention_result["summary"],"assumptions":retention_result["assumptions"]},
    {"draws":retention_result["draws"]})''')
    md("advisor_notes","""
    ### 5.5. Advisor kapasitas dan anggaran
    Advisor memeriksa kebijakan yang sudah dikunci dan setiap ukuran top-k yang memenuhi batas kontak/anggaran. Minimum precision memakai label validation. Kebijakan threshold yang melampaui kapasitas ditandai tidak layak, tanpa pemotongan diam-diam. Tidak menjalankan kampanye selalu kandidat dengan net nol. Advisor hanya menyarankan kandidat bila expected net positif dan batas terpenuhi; ia tidak mengganti threshold audit model. Optimasi pada validation memerlukan konfirmasi di data baru dan pilot.
    """)
    code("form_advisor",'''#@title Form 5 - Advisor kapasitas dan anggaran validation
Batas_kontak = 200 #@param {type:"integer"}
Anggaran_UM = 2500.0 #@param {type:"number"}
Biaya_kontak_UM = 10.0 #@param {type:"number"}
Manfaat_retained_UM = 100.0 #@param {type:"number"}
Keberhasilan_asumsi_persen = 20.0 #@param {type:"number"}
Biaya_tetap_advisor_UM = 500.0 #@param {type:"number"}
Minimum_precision_validation_persen = 0.0 #@param {type:"number"}
advisor=v6_policy_advisor(y_valid,validation_probability,policies,Batas_kontak,Anggaran_UM,Biaya_kontak_UM,
    Manfaat_retained_UM,v6_number(Keberhasilan_asumsi_persen,"Keberhasilan",high=100)/100,
    fixed_cost=Biaya_tetap_advisor_UM,min_precision=v6_number(Minimum_precision_validation_persen,"Precision",high=100)/100,index=X_valid.index)
print(advisor["candidates"].to_string(index=False))
print("Keputusan bersyarat:",advisor["status"],json.dumps(v6_json(advisor["recommended"]),indent=2))
save_live_section("advisor",advisor["inputs"],{"status":advisor["status"],"recommended":advisor["recommended"],
    "data_context":"validation","business_inputs_are_assumptions":True},ARTIFACT_DIR,run.context,
    {"candidates":advisor["candidates"],"top_k_curve":advisor["top_k_curve"]})
print("HTML/PDF dan ZIP diperbarui memakai input advisor ini.")''')
    md("uncertainty_notes","""
    ### 5.6. Rentang parameter dan outcome
    Berbeda dari form 4, rentang uniform pada u, C, dan B juga berubah di setiap draw. Kontak tetap. Laporan memisahkan rentang expected net akibat asumsi parameter dari rentang outcome net yang juga mengandung variasi retained. Proporsi draw positif bergantung pada distribusi asumsi yang dipilih; ia bukan peluang laba terverifikasi.
    """)
    code("form_uncertainty",'''#@title Form 6 - Ketidakpastian asumsi bisnis
Kontak_ketidakpastian = 200 #@param {type:"integer"}
Keberhasilan_min_persen = 10.0 #@param {type:"number"}
Keberhasilan_max_persen = 30.0 #@param {type:"number"}
Biaya_min_UM = 8.0 #@param {type:"number"}
Biaya_max_UM = 12.0 #@param {type:"number"}
Manfaat_min_UM = 80.0 #@param {type:"number"}
Manfaat_max_UM = 120.0 #@param {type:"number"}
Biaya_tetap_ketidakpastian_UM = 500.0 #@param {type:"number"}
Jumlah_draw_ketidakpastian = 2000 #@param {type:"integer"}
uncertainty=v6_uncertain_campaign(validation_probability,Kontak_ketidakpastian,
    [Keberhasilan_min_persen/100,Keberhasilan_max_persen/100],[Biaya_min_UM,Biaya_max_UM],
    [Manfaat_min_UM,Manfaat_max_UM],Biaya_tetap_ketidakpastian_UM,n_draws=Jumlah_draw_ketidakpastian,seed=Seed,index=X_valid.index)
print(json.dumps(v6_json(uncertainty["summary"]),indent=2))
save_live_section("uncertainty",uncertainty["assumptions"],{"results":uncertainty["summary"],"assumptions":uncertainty["assumptions"]},
    ARTIFACT_DIR,run.context,{"draws":uncertainty["draws"]})
print("Rentang skenario pada HTML/PDF dan ZIP diperbarui.")''')
    md("batch_notes","""
    ### 5.7. Kualitas batch dan scoring
    Contoh validation tersedia setelah training. Data stres sintetis sengaja menggeser usia, saldo, dan kategori untuk mendemonstrasikan peringatan. Pilihan CSV saya membaca file yang diunggah ke Files; fitur wajib harus lengkap, kolom header ganda/asing ditolak, dan ID pelanggan disimpan hanya pada hasil lokal. Jangan mengunggah data pribadi ke repo public.

    Pemeriksaan membandingkan missingness, kategori/rentang, serta distribusi fitur terhadap train dan skor terhadap validation. Batas alert adalah heuristik eksplorasi, bukan uji signifikansi. Pergeseran distribusi belum membuktikan penurunan akurasi. Tanpa label baru yang sesuai horizon, akurasi batch tidak dihitung; label opsional hanya untuk audit dan tidak melatih model ulang.
    """)
    code("form_batch",'''#@title Form 7 - Audit batch dan scoring
Sumber_batch = "Contoh validation" #@param ["Contoh validation","Stres sintetis","CSV saya"]
Path_batch_CSV = "synthetic_customers.csv" #@param {type:"string"}
if Sumber_batch=="CSV saya":
    batch=read_source_csv(Path(Path_batch_CSV));batch_context="user_supplied_unverified"
else:
    batch=X_valid.head(min(200,len(X_valid))).copy();batch_context="illustrative_validation"
    if Sumber_batch=="Stres sintetis":
        batch["Age"]=95;batch["Balance"]=900000.;batch["Geography"]="SYNTHETIC_UNKNOWN";batch_context="synthetic_stress_demo"
batch_result=v6_batch_quality(X_train,batch,run.bundle,predict_customers,validation_probability,data_context=batch_context)
print(json.dumps(v6_json(batch_result["summary"]),indent=2))
print(batch_result["features"].to_string(index=False))
save_live_section("batch",{"source":Sumber_batch,"path":Path_batch_CSV if Sumber_batch=="CSV saya" else None},batch_result["summary"],
    ARTIFACT_DIR,run.context,{"features":batch_result["features"],"scores":batch_result["scores"]})
print("Hasil scoring disimpan lokal pada run; ringkasan public hanya agregat.")''')
    md("recommendations","""
    ## 6. Rekomendasi dan pengembangan berikutnya

    1. **Definisikan keputusan:** siapa yang boleh dihubungi, horizon churn, kapasitas harian, consent, biaya kontak, nilai kontribusi, dan batas false positive. Pilih kebijakan pada validation sesuai tujuan; skor tinggi saja belum menjamin manfaat intervensi.
    2. **Perbaiki data:** catat tanggal snapshot, tanggal churn, riwayat transaksi/keluhan/engagement, kanal kontak, dan riwayat penawaran. Audit cakupan, missingness, duplikasi pelanggan, serta status lisensi sumber.
    3. **Validasi baru:** bekukan versi pipeline dan kebijakan, lalu nilai pada periode atau populasi yang belum pernah dipakai. Gunakan time/group split bila beberapa catatan berasal dari pelanggan sama. Laporkan ranking, kalibrasi, kapasitas, interval, dan error per segmen.
    4. **Pilot treatment/control:** randomisasikan pelanggan eligible ke penawaran dan kontrol, tetapkan outcome serta horizon sebelum eksperimen. Ukur selisih churn, biaya incremental, nilai kontribusi, dan efek per segmen. Ukuran sampel memerlukan baseline, minimum detectable effect, serta power yang disepakati.
    5. **Operasional:** simpan versi model/sumber/fitur dan jejak keputusan, batasi akses hasil scoring, serta pantau schema, distribusi, kalibrasi dan outcome saat label tersedia. Jadwal retraining harus dipicu evaluasi yang disetujui, bukan hanya alert distribusi.

    Model uplift atau optimasi penawaran dapat dipertimbangkan setelah tersedia treatment/control yang memadai. Data saat ini belum dapat memilih intervensi yang menyebabkan retensi. Dashboard dan API scoring sebaiknya memakai modul inferensi yang sama serta versi bundle yang terverifikasi.
    """)
    md("export_notes","""
    ## 7. Pemeriksaan dan ekspor

    Run dianggap complete setelah verifikasi split, prediksi reload, helper inferensi mandiri, metrik, laporan, dan arsip berhasil. `latest_run.json` menunjuk run terakhir yang lengkap. HTML bersifat mandiri, PDF merangkum hasil dan asumsi, `public_summary.json` berisi agregat, sedangkan ZIP juga memuat model dan hasil batch lokal sehingga harus ditinjau sebelum dibagikan.

    Ubah form lalu unduh ZIP terbaru tanpa training ulang. Tautan di bawah bisa dipakai di Jupyter; di Colab buka panel **Files > bank_churn_artifacts_v7**, pilih PDF/HTML dalam folder run atau ZIP pada root folder tersebut, lalu Download.
    """)
    code("export",'''manifest=finish_run(run,initialize_examples=False)
print(json.dumps(v6_json(manifest),indent=2))
print("Pemeriksaan:",len(run.checks),"PASS")
for item in run.checks:print("PASS -",item["check"])
for filename in ["executive_summary.pdf","executive_summary.html","public_summary.json","current_report.md"]:
    display(FileLink(str(ARTIFACT_DIR/filename)))
display(FileLink(manifest["archive"]))
print("ZIP memuat bundle dan scoring lokal. Public repo hanya memakai ringkasan agregat yang ditinjau.")''')
    md("references","""
    ## Referensi metode dan catatan versi

    - [scikit-learn: pemilihan threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
    - [scikit-learn: kalibrasi probabilitas](https://scikit-learn.org/stable/modules/calibration.html)
    - [scikit-learn: nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
    - [scikit-learn: permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
    - [uv: lockfile dan lingkungan](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile)

    V7 mengunci dependensi, menyatukan training/inference, menambah uji alur lengkap sintetis dan laporan HTML/PDF. Perubahan arsitektur tidak otomatis meningkatkan performa prediktif. Angka run ini selalu dihitung kembali dari sumber dan konfigurasi yang dicatat. V6 dan notebook asli tetap tersedia terpisah.
    """)
    for cell in cells:
        if cell.cell_type=="code" and cell.source.startswith("#@title"):
            cell.metadata["cellView"]="form"
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"},"colab":{"name":"Banking Churn Reduction - V7.ipynb","provenance":[]},
        "banking_churn":{"version":7,"package_sha256":hashes,"output_policy":"public_source_without_outputs"}})
    nbf.validate(nb)
    target=ROOT/"notebooks/banking_churn_v7.ipynb";nbf.write(nb,target)
    print("Built",target.name,":",sum(c.cell_type=="code" for c in cells),"code cells,",sum(c.cell_type=="markdown" for c in cells),"explanation cells")
    return nb


if __name__=="__main__":build()
