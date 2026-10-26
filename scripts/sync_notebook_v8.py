"""Build a separate V8 supplement; do not rewrite the preserved V7 training notebook."""
import ast
import hashlib
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__.py","schema.py","inference.py","synthetic.py","v8_validation.py","v8_pilot.py")


def build():
    cells = []
    def md(key,source):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(),id=key))
    def code(key,source):cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(),id=key))
    md("intro",r"""
    # Banking Churn Reduction — V8: Audit Model Beku & Rencana Pilot

    V8 melanjutkan V7 dengan **validasi, ketidakpastian dan rancangan eksperimen**. Model V7 serta notebook yang sudah berisi hasil tetap disimpan terpisah. Tidak ada training ulang atau pencarian threshold di notebook ini.

    **Mulai:** Runtime > Run all. Default memakai data dan aturan skor fiktif agar semua form dapat dicoba tanpa unduhan CSV pelanggan. Untuk model sendiri pilih `bundle_tepercaya`, unggah bundle V7 beserta CSV development, dan isi path. Bundle joblib harus berasal dari run Anda sendiri yang tepercaya; format ini dapat menjalankan kode saat dimuat. Gunakan versi dependensi yang sama dengan training.

    | Pertanyaan | Jawaban V8 |
    |---|---|
    | Apakah data benar-benar baru? | Cek hash file, overlap ID dan baris fitur terhadap **seluruh** CSV development, bukan hanya train. |
    | Seberapa pasti precision/recall dan AUC/AP? | Interval Wilson untuk proporsi, paired row bootstrap untuk metrik skor. |
    | Apa yang dilakukan sebelum operasional? | Lengkapi definisi target, horizon, kematangan label, provenance dan review independen. |
    | Berapa pelanggan untuk uji retensi? | Rencana dua kelompok acak dengan efek, power dan kehilangan outcome sebagai asumsi. |

    Data baru tanpa overlap masih dapat memiliki leakage atau masalah pemilihan sampel. Metadata sumber/periode adalah **deklarasi pengguna**, bukan bukti yang disahkan software. Notebook tidak pernah menamai hasil sebagai validasi eksternal yang terverifikasi hanya karena sebuah mode dipilih.
    """)
    md("environment", """
    ## 1. Setup dan reproduksi

    Python 3.12/3.13 dan lock analisis yang sama dengan V7. Kernel/UI Colab dipertahankan. Jika setup mengganti paket yang sudah dimuat, pilih Runtime > Restart session lalu Run all. Mode demo tidak melatih model. Versi aktual, hash bundle, hash CSV dan kebijakan yang dipakai dicatat pada hasil.
    """)
    v7 = nbf.read(ROOT/"notebooks/banking_churn_v7.ipynb",as_version=4)
    deps = next(c for c in v7.cells if c.id=="dependencies")
    cells.append(nbf.v4.new_code_cell(deps.source.replace("V7 memerlukan","V8 memerlukan"),id="dependencies"))
    sources = {name:(ROOT/"src/banking_churn"/name).read_text(encoding="utf-8") for name in MODULES}
    hashes = {name:hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}
    bootstrap = "PACKAGE_SOURCES = "+repr(sources)+"\nPACKAGE_SHA256 = "+repr(hashes)+"\n"+textwrap.dedent('''
        import hashlib,importlib,json,sys,tempfile
        from pathlib import Path
        package_root=Path(tempfile.mkdtemp(prefix="bank-churn-v8-"))
        package_dir=package_root/"banking_churn";package_dir.mkdir()
        for name,source in PACKAGE_SOURCES.items():
            assert hashlib.sha256(source.encode()).hexdigest()==PACKAGE_SHA256[name]
            (package_dir/name).write_text(source,encoding="utf-8")
        for name in list(sys.modules):
            if name=="banking_churn" or name.startswith("banking_churn."):del sys.modules[name]
        sys.path.insert(0,str(package_root));importlib.invalidate_caches()
        import numpy as np,pandas as pd,matplotlib.pyplot as plt,joblib
        from IPython.display import display,Markdown,FileLink
        from banking_churn.v8_validation import ValidationEvidence,read_validation_csv,build_reference_profile,audit_frozen_model,demo_bundle
        from banking_churn.v8_pilot import PilotPlan,simulate_pilot,approximate_power
        from banking_churn.synthetic import make_synthetic_data
        from banking_churn.schema import FEATURES
        print("V8: enam modul terverifikasi; model dan threshold tidak di-fit ulang.")
    ''').strip()
    code("shared_package",bootstrap)
    md("data_protocol", """
    ## 2. Data dan protokol yang perlu dibekukan

    **Development** mencakup train, validation dan test lama. Menggunakan test lama lagi hanya menghasilkan audit ulang. Isikan CSV development yang hash-nya cocok dengan metadata bundle agar V8 dapat memeriksa overlap. Jika CSV tidak tersedia, scoring tetap dapat dilakukan tetapi pemeriksaan independensi dinyatakan belum lengkap.

    **Batch baru** hanya menerima sepuluh fitur V7, `CustomerId` dan `Exited` opsional. ID dibaca sebagai teks agar nol di depan tetap ada; ID kosong/duplikat setelah whitespace dibersihkan ditolak. Target wajib 0/1 lengkap bila kolomnya disertakan. Jangan mengisi label yang belum matang sebagai nol. Tanpa label matang, metrik prediksi ditunda.

    **Definisi yang perlu ditulis sebelum melihat hasil:** unit pelanggan, populasi, periode observasi, definisi churn, panjang horizon (misalnya 90 hari), tanggal akhir follow-up, kebijakan kontak tetap, dan kriteria penerimaan yang relevan bagi operasi. Jangan memilih threshold baru berdasarkan audit ini lalu mengklaim kinerja yang sama sebagai estimasi independen.

    Pada mode bundle tepercaya, `Path_Batch_Baru` yang kosong sengaja menjalankan **batch sintetis dengan model Anda**. Ini menguji alur scoring, bukan menilai kinerja model pada populasi baru. CSV sumber V7 belum memiliki horizon terverifikasi; isi metadata hanya setelah bukti tersedia.
    """)
    code("configuration",'''#@title 2.1 - Sumber dan bukti validasi
Mode = "demo_sintetis" #@param ["demo_sintetis", "bundle_tepercaya"]
Path_Bundle = "bank_churn_bundle.joblib" #@param {type:"string"}
Path_Referensi = "Churn_Modelling.csv" #@param {type:"string"}
Path_Batch_Baru = "" #@param {type:"string"}
Deskripsi_sumber = "" #@param {type:"string"}
Populasi = "" #@param {type:"string"}
Periode_observasi = "" #@param {type:"string"}
Horizon_hari = 0 #@param {type:"integer"}
Definisi_target_dikonfirmasi = False #@param {type:"boolean"}
Label_sudah_matang = False #@param {type:"boolean"}
Independensi_ditinjau = False #@param {type:"boolean"}
OUTPUT_BASE = Path("bank_churn_artifacts_v8")
import uuid
RUN_ID="v8-"+uuid.uuid4().hex[:12]
RUN_DIR=OUTPUT_BASE/RUN_ID;RUN_DIR.mkdir(parents=True,exist_ok=False)
print("Folder hasil baru:",RUN_DIR)''')
    code("load_frozen",'''BUNDLE_PATH=None
if Mode=="demo_sintetis":
    reference=make_synthetic_data(1000,42)
    reference_hash=hashlib.sha256(reference.to_csv(index=False).encode()).hexdigest()
    bundle=demo_bundle(reference,reference_hash)
    profile=build_reference_profile(reference,bundle,reference_hash)
elif Mode=="bundle_tepercaya":
    BUNDLE_PATH=Path(Path_Bundle)
    BUNDLE_HASH=hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
    bundle=joblib.load(BUNDLE_PATH)
    reference=None;profile=None
    if Path_Referensi.strip():
        ref_path=Path(Path_Referensi)
        reference=read_validation_csv(ref_path)
        reference_hash=hashlib.sha256(ref_path.read_bytes()).hexdigest()
        profile=build_reference_profile(reference,bundle,reference_hash)
else:raise ValueError("Mode tidak dikenal")
if Path_Batch_Baru.strip():
    batch_path=Path(Path_Batch_Baru)
    incoming=read_validation_csv(batch_path)
    incoming_hash=hashlib.sha256(batch_path.read_bytes()).hexdigest()
    IS_SYNTHETIC=Mode=="demo_sintetis"
else:
    incoming=make_synthetic_data(1000,123).loc[:,FEATURES+["CustomerId","Exited"]]
    incoming["CustomerId"]="SYNTH-NEW-"+incoming.CustomerId.astype(str)
    incoming_hash=hashlib.sha256(incoming.to_csv(index=False).encode()).hexdigest()
    IS_SYNTHETIC=True
evidence=ValidationEvidence(Deskripsi_sumber,Populasi,Periode_observasi,Horizon_hari,
    Definisi_target_dikonfirmasi,Label_sudah_matang,Independensi_ditinjau)
if IS_SYNTHETIC:
    evidence=ValidationEvidence("Generator fiktif untuk demonstrasi","Pelanggan sintetis","Tidak ada periode nyata",90,True,True,False)
POLICY_HASH=hashlib.sha256(json.dumps(bundle["metadata"]["policies"],sort_keys=True).encode()).hexdigest()
print("DATA SINTETIS - uji alur saja" if IS_SYNTHETIC else "CSV BARU - provenance dan independensi perlu ditinjau")
print("Model run:",bundle["metadata"]["run_id"],"| Kebijakan:",bundle["metadata"]["default_policy"])
print("Hash CSV batch:",incoming_hash)
print("Tidak ada training, kalibrasi atau optimasi threshold pada V8.")''')
    md("audit_notes", r"""
    ## 3. Audit skor dan ketidakpastian

    Jalankan form berikut tanpa mengubah model. ROC-AUC membaca pemeringkatan; average precision (AP) bergantung pula pada prevalensi; Brier/log loss menilai kualitas probabilitas. Precision dan recall bergantung pada kebijakan kontak yang sudah tersimpan. **Bandingkan populasi dan prevalensi sebelum membandingkan AP antar-dataset.**

    Precision/recall, contact rate dan prevalensi memakai interval **Wilson 95%**, termasuk kasus nol kejadian. Denominator nol berarti tidak terdefinisi, bukan nilai nol. AUC dan AP tidak dilaporkan bila label hanya satu kelas. Bootstrap berpasangan mengambil ulang baris label/skor bersama-sama; interval ini mengasumsikan observasi independen dan tidak mencakup seluruh ketidakpastian pemilihan model atau perubahan populasi. Untuk pelanggan berulang, cabang bank, atau rangkaian waktu, gunakan desain cluster/time yang sesuai.

    Status `ready_for_independent_review` berarti daftar bukti yang diminta telah diisi, pemeriksaan overlap terlewati dan tidak ada alarm yang dikenali. Itu **bukan sertifikasi validasi eksternal** atau persetujuan deployment. Missing, kategori asing dan nilai di luar rentang perlu ditinjau; alarm data tidak otomatis membuktikan performa menurun.
    """)
    code("audit_form",'''#@title 3.1 - Audit model beku
Bootstrap_resamples = 200 #@param {type:"integer"}
Seed_audit = 42 #@param {type:"integer"}
audit=audit_frozen_model(incoming,bundle,evidence,profile,incoming_hash,IS_SYNTHETIC,Bootstrap_resamples,Seed_audit)
IS_SYNTHETIC=audit["summary"]["synthetic"]
display(Markdown("**Status: "+audit["summary"]["status"]+"**"))
display(pd.DataFrame({"rekomendasi":audit["summary"]["recommendations"]}))
display(audit["metrics"])
display(audit["quality"])
if not audit["calibration"].empty:
    cal=audit["calibration"]
    fig,ax=plt.subplots(figsize=(6,4));ax.plot([0,1],[0,1],"--",color="grey",label="Garis referensi")
    ax.plot(cal.mean_estimated_risk,cal.observed_rate,"o-",label="Batch yang diaudit")
    ax.set(xlabel="Rata-rata probabilitas per bin",ylabel="Proporsi label churn",title=("DATA SINTETIS: " if IS_SYNTHETIC else "")+"kalibrasi deskriptif")
    ax.legend();plt.show();display(cal)
print("Skor pelanggan hanya diekspor secara lokal; ringkasan tidak memuat baris pelanggan.")''')
    md("overlap_notes", """
    ## 4. Simulasi kegagalan independensi

    Form ini sengaja mengirim sebagian CSV development kembali ke audit. Hash file dapat berbeda karena subset atau urutan baris berubah, tetapi ID/baris fitur masih cocok. Hasil harus tetap meminta review. Coba juga mengubah ID menjadi duplikat: scoring harus diblokir tanpa skor parsial. Identitas yang berubah dan fitur yang ikut berubah dapat lolos pencocokan; **nol overlap bukan bukti independensi**.

    Pemeriksaan ini menutup keterbatasan audit batch lama yang menamai audit sebagai eksternal dari pilihan mode `external_csv`. V8 selalu mencatat `external_validation_verified=False`; analis perlu memeriksa provenance, waktu, unit observasi dan proses pembuatan label.
    """)
    code("overlap_form",'''#@title 4.1 - Uji data berulang atau ID tidak valid
Skenario_overlap = "subset_development" #@param ["subset_development", "duplicate_id"]
if reference is None:
    overlap_demo={"summary":{"status":"reference_unavailable","recommendations":["Sediakan CSV development yang hash-nya cocok untuk simulasi overlap."]}}
else:
    columns=FEATURES+[c for c in ["CustomerId","Exited"] if c in reference]
    repeated=reference.loc[:,columns].head(50).copy()
    if Skenario_overlap=="duplicate_id":
        repeated["CustomerId"]="DUPLICATE"
    overlap_demo=audit_frozen_model(repeated,bundle,evidence,profile,
        hashlib.sha256(repeated.to_csv(index=False).encode()).hexdigest(),IS_SYNTHETIC,20,42)
display(Markdown("**Hasil simulasi overlap: "+overlap_demo["summary"]["status"]+"**"))
overlap_summary=overlap_demo["summary"]
display(pd.DataFrame({"pemeriksaan":["Status","ID overlap","Baris fitur overlap","Validasi eksternal disahkan software"],
    "hasil":[overlap_summary["status"],overlap_summary.get("overlap_customer_ids"),overlap_summary.get("overlap_unique_feature_rows"),False]}))
display(pd.DataFrame({"rekomendasi":overlap_summary["recommendations"]}))''')
    md("pilot_notes", r"""
    ## 5. Rencana pilot retensi dan simulasi power

    Churn berisiko tinggi tidak otomatis berarti pelanggan akan terbantu oleh penawaran. Gunakan uji acak: pelanggan yang memenuhi kriteria dibagi **1:1** ke penawaran dan kontrol, dengan definisi outcome dan horizon yang sama. Bekukan unit randomisasi, perlakuan, outcome utama, tanggal analisis dan aturan penghentian sebelum eksperimen. Analisis utama mengikuti kelompok penugasan awal (*intention to treat*).

    Form memakai churn kontrol 20% dan target penurunan 3 **poin persentase** menjadi 17% sebagai contoh. Penurunan relatifnya 15%. Seluruh efek merupakan **asumsi perencanaan**, bukan keluaran model churn. Power 80% berarti peluang desain mendeteksi efek asumsi saat efek itu benar, bukan probabilitas penawaran pasti berhasil.

    Perencanaan memakai aproksimasi normal dua sisi dengan Cohen's $h=2\arcsin\sqrt{p_0}-2\arcsin\sqrt{p_1}$, alokasi seimbang, satu outcome dan observasi independen. Ukuran yang dapat dianalisis dihitung dahulu, lalu jumlah rekrutmen dinaikkan menurut asumsi outcome hilang. Inflasi sampel tidak menghilangkan bias kehilangan follow-up. Normal approximation, clustering, pengujian berulang, noncompliance dan banyak outcome perlu peninjauan desain lebih lanjut.

    Simulasi Monte Carlo membangkitkan outcome binomial dan menjalankan uji dua proporsi dua sisi. Karena uji simulasi memakai statistik pooled-z dan perencanaan memakai transformasi arcsine, hasilnya dapat sedikit berbeda. Interval di samping rejection rate menunjukkan **error Monte Carlo**, bukan interval dampak bisnis nyata. Simulasi nol efek memperlihatkan peluang false positive mendekati alpha.
    """)
    code("pilot_form",'''#@title 5.1 - Kebutuhan sampel dan eksperimen fiktif
Churn_kontrol = 0.20 #@param {type:"number"}
Penurunan_absolut = 0.03 #@param {type:"number"}
Target_power = 0.80 #@param {type:"number"}
Alpha = 0.05 #@param {type:"number"}
Outcome_hilang = 0.10 #@param {type:"number"}
Jumlah_simulasi = 5000 #@param {type:"integer"}
Seed_pilot = 42 #@param {type:"integer"}
plan=PilotPlan(Churn_kontrol,Penurunan_absolut,Target_power,Alpha,Outcome_hilang)
pilot=plan.calculate()
alternative,draws=simulate_pilot(plan,repetitions=Jumlah_simulasi,seed=Seed_pilot)
null_summary,null_draws=simulate_pilot(plan,repetitions=Jumlah_simulasi,seed=(Seed_pilot+1)%(2**32),null_effect=True)
display(pd.DataFrame({"parameter":["Churn kontrol (asumsi)","Churn penawaran (asumsi)","Penurunan (poin persentase)",
    "Outcome tersedia per kelompok","Rekrutmen per kelompok","Rekrutmen total","Power aproksimasi","Alpha dua sisi"],
    "nilai":[pilot["control_rate_assumption"],pilot["treatment_rate_assumption"],pilot["reduction_percentage_points"],
    pilot["analyzable_per_arm"],pilot["enrolment_per_arm"],pilot["total_enrolment"],pilot["approximate_power"],pilot["alpha"]]}))
display(pd.DataFrame([alternative,null_summary])[["null_effect_simulation","analyzable_per_arm","rejection_rate","monte_carlo_interval_95"]])
sensitivity=[]
for effect in [.01,.02,.03,.05]:
    if effect<Churn_kontrol:
        item=PilotPlan(Churn_kontrol,effect,Target_power,Alpha,Outcome_hilang).calculate()
        sensitivity.append({"target_penurunan_pp":100*effect,"rekrutmen_total":item["total_enrolment"]})
display(pd.DataFrame(sensitivity))
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(draws.observed_reduction_pp,bins=35,alpha=.75,label="Efek asumsi")
axes[0].hist(null_draws.observed_reduction_pp,bins=35,alpha=.55,label="Nol efek")
axes[0].set(xlabel="Penurunan churn teramati (poin persentase)",ylabel="Eksperimen fiktif",title="Variasi hasil karena sampling");axes[0].legend()
n_grid=np.linspace(max(2,pilot["analyzable_per_arm"]//4),2*pilot["analyzable_per_arm"],30).astype(int)
axes[1].plot(n_grid,[approximate_power(Churn_kontrol,Churn_kontrol-Penurunan_absolut,int(n),Alpha) for n in n_grid])
axes[1].axhline(Target_power,color="grey",linestyle="--")
axes[1].set(xlabel="Outcome yang tersedia per kelompok",ylabel="Power aproksimasi",ylim=(0,1),title="Efek asumsi dan ukuran sampel")
plt.tight_layout();plt.show()
print("Rekrutmen total:",pilot["total_enrolment"],"| Bukan klaim efek retensi atau ROI nyata.")''')
    md("recommendations", """
    ## 6. Rekomendasi praktis dan urutan kerja

    1. **Sebelum data baru:** konfirmasi arti `Exited`, populasi, periode dan horizon; mintakan outcome matang dan pengenal stabil. Simpan snapshot model/kebijakan serta hash sumber.
    2. **Saat data baru datang:** jalankan audit beku, tinjau overlap dan kualitas fitur, lalu baca AUC/AP, kalibrasi, precision/recall beserta denominator dan intervalnya. Cek segmen operasional yang relevan secara terpisah; aggregate yang baik belum menjamin setiap segmen baik.
    3. **Sebelum pilot:** tentukan efek minimum yang berguna, biaya per kontak, kapasitas dan anggaran. Pastikan jumlah eligible customers cukup untuk dua kelompok; bila belum cukup, tambah periode rekrutmen atau revisi desain secara eksplisit.
    4. **Saat pilot:** randomisasi sebelum kontak, catat penugasan, penawaran, penerimaan, biaya dan outcome pada horizon yang disepakati. Gunakan kontrol untuk mengukur efek incremental; angka risiko model bukan pengganti kontrol.
    5. **Setelah pilot:** laporkan selisih churn dan biaya beserta ketidakpastian, kehilangan outcome dan pelanggaran protokol. Pertimbangkan uplift hanya setelah ada data intervensi yang memadai.

    Hasil saat ini dapat membuktikan software berjalan dan menunjukkan celah bukti. Tanpa data periode/populasi baru yang diperiksa dan pilot terkontrol, belum ada dasar baru untuk menyatakan model siap operasional atau program retensi menghasilkan profit.

    File skor pelanggan disimpan **lokal/private**. Ringkasan audit memuat statistik dan deklarasi sumber yang Anda isi; tinjau juga teks metadata sebelum membagikannya. V8 tidak menyalin bundle model maupun CSV sumber ke repo public.
    """)
    code("export",'''import html,importlib.metadata as package_metadata,platform
def save_json(name,value):
    (RUN_DIR/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\\n",encoding="utf-8")
save_json("audit_summary.json",audit["summary"])
save_json("overlap_demo_summary.json",overlap_demo["summary"])
save_json("pilot_plan.json",pilot)
save_json("pilot_simulations.json",{"alternative":alternative,"null":null_summary})
for name in ["metrics","quality","calibration","scores"]:
    audit[name].to_csv(RUN_DIR/(name+".csv"),index=name=="scores")
note="\\n".join("- "+item for item in audit["summary"]["recommendations"])
(RUN_DIR/"recommendations.md").write_text("# Rekomendasi V8\\n\\n"+note+"\\n\\nPilot memakai efek asumsi, bukan dampak terukur.\\n",encoding="utf-8")
banner="DATA SINTETIS - demonstrasi software" if IS_SYNTHETIC else "AUDIT DATA - perlu review independen"
body="<h1>V8: audit model beku dan rencana pilot</h1><h2>"+banner+"</h2><p>Efek pilot adalah asumsi. Belum ada validasi eksternal yang disahkan software.</p>"
body+="<h2>Audit</h2><pre>"+html.escape(json.dumps(audit["summary"],ensure_ascii=False,indent=2))+"</pre>"
body+=audit["metrics"].to_html(index=False,escape=True)+audit["quality"].to_html(index=False,escape=True)
body+="<h2>Rencana pilot</h2>"+pd.DataFrame({"parameter":list(pilot),"nilai":[str(v) for v in pilot.values()]}).to_html(index=False,escape=True)
body+="<h2>Simulasi fiktif</h2>"+pd.DataFrame([alternative,null_summary]).to_html(index=False,escape=True)
(RUN_DIR/"v8_summary.html").write_text('<!doctype html><html lang="id"><meta charset="utf-8"><title>Banking Churn V8</title><style>body{font:15px system-ui;margin:2rem;color:#18324d}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:.5rem;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'+body+"</html>",encoding="utf-8")
assert POLICY_HASH==hashlib.sha256(json.dumps(bundle["metadata"]["policies"],sort_keys=True).encode()).hexdigest()
if BUNDLE_PATH is not None:assert BUNDLE_HASH==hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
manifest={"notebook_version":8,"run_id":RUN_ID,"status":"complete","synthetic_batch":IS_SYNTHETIC,
    "model_run_id":bundle["metadata"]["run_id"],"model_bundle_sha256":BUNDLE_HASH if BUNDLE_PATH else None,
    "model_refit":False,"policy_unchanged":True,"audit_status":audit["summary"]["status"],
    "python":platform.python_version(),"versions":{p:package_metadata.version(p) for p in ["numpy","pandas","scikit-learn","joblib"]}}
save_json("v8_manifest.json",manifest)
print(json.dumps(manifest,indent=2))
for name in ["v8_summary.html","recommendations.md","audit_summary.json","pilot_plan.json"]:
    display(FileLink(str(RUN_DIR/name)))
print("Selesai. scores.csv adalah private; periksa metadata sebelum membagikan ringkasan.")''')
    md("references", """
    ## Referensi metode

    - [NIST: interval Wilson untuk proporsi](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
    - [statsmodels: normal power dua sampel independen](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.solve_power.html)
    - [statsmodels: effect size dua proporsi](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_effectsize.html)
    - [scikit-learn: pentingnya data evaluasi yang terpisah](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

    Implementasi V8 memakai NumPy, scikit-learn dan Python `NormalDist`; statsmodels dicantumkan sebagai referensi metode dan tidak menambah dependensi. V7 tetap menjadi sumber training; V8 adalah suplemen audit dan perencanaan, bukan model baru yang lebih akurat.
    """)
    for cell in cells:
        if cell.cell_type=="code" and cell.source.startswith("#@title"):cell.metadata["cellView"]="form"
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"},"colab":{"name":"Banking Churn Reduction - V8 Audit & Pilot.ipynb","provenance":[]},
        "banking_churn":{"version":8,"package_sha256":hashes,"output_policy":"public_source_without_outputs"}})
    nbf.validate(nb)
    target=ROOT/"notebooks/banking_churn_v8.ipynb";nbf.write(nb,target)
    print("Built",target.name,":",sum(c.cell_type=="code" for c in cells),"code cells,",sum(c.cell_type=="markdown" for c in cells),"explanation cells")
    return nb


if __name__=="__main__":build()
