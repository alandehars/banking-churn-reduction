"""Self-contained aggregate HTML/PDF reports; no customer rows or fitted model publication."""
import base64
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image,PageBreak,Paragraph,SimpleDocTemplate,Spacer,Table,TableStyle

NAVY="#18324D"
TEAL="#138B8D"
AMBER="#A75E13"
RED="#AD3C37"


def plot_model_audit(run):
    y=run.labels["test"];p=run.probabilities["test"]
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.45))
    RocCurveDisplay.from_predictions(y,p,ax=axes[0],curve_kwargs={"color":TEAL})
    axes[0].plot([0,1],[0,1],"--",color="#87939D",linewidth=1)
    PrecisionRecallDisplay.from_predictions(y,p,ax=axes[1],curve_kwargs={"color":TEAL})
    axes[1].axhline(float(y.mean()),linestyle="--",color="#87939D",label="Prevalensi test")
    for axis in axes:
        axis.grid(alpha=.17);axis.spines[["top","right"]].set_visible(False)
        axis.tick_params(labelsize=8);axis.legend(fontsize=7)
    axes[0].set_title("Ranking risiko - test",fontsize=11,color=NAVY)
    axes[1].set_title("Precision-recall - test",fontsize=11,color=NAVY)
    badge="DATA SINTETIS / UJI ALUR" if run.metadata["synthetic"] else "DATA REFERENSI / AUDIT INTERNAL"
    fig.suptitle(badge+" | "+run.metadata["model_name"],fontsize=11,fontweight="bold",color=NAVY)
    fig.tight_layout(rect=(0,0,1,.93))
    target=run.artifact_dir/"model_audit.png"
    fig.savefig(target,dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return target


def _load(path,default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def aggregate_summary(directory):
    directory=Path(directory)
    meta=_load(directory/"model_metadata.json")
    if not meta:raise ValueError("Model metadata is required before report export")
    policies=pd.read_csv(directory/"test_policies.csv")
    current=policies.loc[policies.policy==meta["default_policy"]].iloc[0]
    sections={}
    for kind in ["advisor","uncertainty","comparison","batch"]:
        item=_load(directory/f"v6_{kind}_latest.json")
        if item:
            if item["run_id"]!=meta["run_id"] or item["source_sha256"]!=meta["data_sha256"]:
                raise ValueError("Cross-run snapshot cannot enter a report")
            # Include only aggregate results and business assumptions, never row-level scores.
            sections[kind]={"inputs":item["inputs"] if kind in {"advisor","uncertainty"} else {},"summary":item["summary"]}
    summary={"report_version":7,"run_id":meta["run_id"],"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "data_context":meta["data_context"],"synthetic":meta["synthetic"],
        "dataset_rows":meta["rows_used"],"source_sha256":meta["data_sha256"],
        "source_reference_matches":meta["source_reference_matches"],"model":meta["model_name"],
        "test_scores":meta["test_scores"],"default_policy":meta["default_policy"],"threshold":meta["threshold"],
        "default_metrics":{key:int(current[key]) if key in {"contacts","tp","fp","fn","tn"} else float(current[key])
            for key in ["contacts","tp","fp","fn","tn","precision","recall","f1","contact_rate"]},
        "test_policies":policies[["policy","contacts","tp","fp","fn","precision","recall"]].to_dict(orient="records"),
        "config":meta["config"],"package_version":meta["package_version"],"versions":meta["versions"],
        "split_fingerprints":meta["split_fingerprints"],"latest_sections":sections,
        "limits":meta["limits"],"customer_records_included":False,"causal_campaign_effect_estimated":False}
    return summary


def _fmt(value,digits=2):
    return "-" if value is None else f"{value:,.{digits}f}"


def _assumption_rows(summary):
    section=summary["latest_sections"].get("advisor")
    if not section:return []
    inp=section["inputs"]
    return [["Batas kontak",str(inp["max_contacts"])],["Anggaran",_fmt(inp["budget_um"])+" UM"],
        ["Biaya per kontak C",_fmt(inp["contact_cost_um"])+" UM"],
        ["Manfaat per retained B",_fmt(inp["value_per_saved_um"])+" UM"],
        ["Keberhasilan u",f"{inp['success_rate_assumption']:.1%} (asumsi)"],
        ["Biaya tetap F",_fmt(inp["fixed_cost_um"])+" UM"],
        ["Minimum precision validation",f"{inp['min_validation_precision']:.1%}"]]


def _decision_text(summary):
    section=summary["latest_sections"].get("advisor")
    if not section:return "Belum ada skenario kebijakan."
    a=section["summary"];best=a["recommended"]
    if a["status"]!="conditional_candidate_found":
        return "Tidak ada kampanye bernilai positif yang memenuhi batas input. Pilihan skenario: tidak menjalankan kampanye."
    return (f"Kandidat bersyarat {best['policy']}: {best['contacts']} kontak, biaya {_fmt(best['cost_um'])} UM, "
            f"expected net {_fmt(best['expected_net_um'])} UM. Ini hasil perhitungan asumsi, bukan laba terukur.")


def _uncertainty_text(summary):
    section=summary["latest_sections"].get("uncertainty")
    if not section:return "Belum ada skenario rentang parameter."
    s=section["summary"]["results"];a=section["summary"]["assumptions"]
    return (f"{s['contacts']} kontak tetap. Rentang uniform u {a['success_uniform'][0]:.0%}-{a['success_uniform'][1]:.0%}, "
        f"C {_fmt(a['cost_uniform_um'][0])}-{_fmt(a['cost_uniform_um'][1])} UM, "
        f"B {_fmt(a['benefit_uniform_um'][0])}-{_fmt(a['benefit_uniform_um'][1])} UM; "
        f"F {_fmt(a['fixed_cost_um'])} UM hanya bila ada kontak. "
        f"Kuantil outcome net 2,5%-97,5%: {_fmt(s['outcome_net_q025_um'])} sampai {_fmt(s['outcome_net_q975_um'])} UM; "
        f"proporsi draw positif {s['profit_fraction_under_assumptions']:.1%}. "
        "Distribusi parameter adalah asumsi eksplorasi; ketidakpastian kalibrasi dan perubahan populasi belum tercakup.")


def _ascii(text):
    for before,after in {"\u2013":"-","\u2014":"-","\u2011":"-","\u00d7":"x","\u2264":"<=","\u2265":">=","\u03a3":"Sum"}.items():
        text=str(text).replace(before,after)
    return str(text)


def _pdf(summary,directory):
    target=directory/"executive_summary.pdf"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand",fontName="Helvetica-Bold",fontSize=10,textColor=colors.HexColor(TEAL),spaceAfter=8))
    styles.add(ParagraphStyle(name="Lead",fontName="Helvetica-Bold",fontSize=23,leading=27,textColor=colors.HexColor(NAVY),spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallBody",fontName="Helvetica",fontSize=9,leading=13,spaceAfter=7,textColor=colors.HexColor(NAVY)))
    styles.add(ParagraphStyle(name="TinyNote",fontName="Helvetica",fontSize=7,leading=9,textColor=colors.HexColor("#5D6B76")))
    styles.add(ParagraphStyle(name="TableCell",fontName="Helvetica",fontSize=8,leading=11,textColor=colors.HexColor(NAVY)))
    styles.add(ParagraphStyle(name="SectionHead",fontName="Helvetica-Bold",fontSize=13,leading=17,textColor=colors.HexColor(NAVY),spaceBefore=9,spaceAfter=7))
    P=lambda text,style="SmallBody":Paragraph(html.escape(_ascii(text)),styles[style])
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=17*mm,
        title="Banking Churn V7 - Executive Summary",author="Banking Churn Reduction project")
    badge="DATA SINTETIS - UJI ALUR, BUKAN HASIL BANK NYATA" if summary["synthetic"] else "DATA REFERENSI - AUDIT INTERNAL"
    story=[P("BANKING CHURN REDUCTION / V7","Brand"),P("Ringkasan hasil dan keputusan","Lead"),P(badge,"Brand")]
    d=summary["default_metrics"];m=summary["test_scores"]
    story.append(P(f"Model {summary['model']} | Test {m['test_rows']:,} catatan | {m['test_churn']:,} label churn"))
    cards=[]
    for label,value in [("ROC-AUC",f"{m['roc_auc']:.3f}"),("Average precision",f"{m['average_precision']:.3f}"),
                        ("Precision default",f"{d['precision']:.1%}"),("Recall default",f"{d['recall']:.1%}")]:
        cards.append([Paragraph(label,styles["TableCell"]),Spacer(1,3*mm),
            Paragraph(f'<font size="20"><b>{value}</b></font>',
                ParagraphStyle("CardValue",fontName="Helvetica",fontSize=20,leading=24,textColor=colors.HexColor(NAVY)))])
    cardtable=Table([cards],colWidths=[43.5*mm]*4)
    cardtable.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EDF5F6")),("BOX",(0,0),(-1,-1),.4,colors.HexColor("#D9E5E8")),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.extend([cardtable,Spacer(1,9*mm)])
    story.append(P(f"Default {summary['default_policy']} memakai threshold {summary['threshold']:.6f} dari validation. "
        f"Dari {d['contacts']} prioritas, terdapat {d['tp']} label churn terdeteksi, {d['fp']} alarm keliru, dan {d['fn']} kasus terlewat. "
        "Deteksi churn belum mengukur pelanggan yang terselamatkan."))
    if (directory/"model_audit.png").is_file():
        story.append(Image(str(directory/"model_audit.png"),width=174*mm,height=61*mm,kind="proportional"))
    story.append(P("Tradeoff kebijakan yang sudah dikunci","SectionHead"))
    header=["Kebijakan","Kontak","TP","FP","FN","Precision","Recall"]
    rows=[[P(c,"TableCell") for c in header]]
    for r in summary["test_policies"]:
        rows.append([P(str(r["policy"]),"TableCell"),str(int(r["contacts"])),str(int(r["tp"])),str(int(r["fp"])),str(int(r["fn"])),f"{r['precision']:.1%}",f"{r['recall']:.1%}"])
    table=Table(rows,colWidths=[45*mm,20*mm,16*mm,16*mm,16*mm,30*mm,31*mm],repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5EEF1")),("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor(TEAL)),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,1),(-1,-1),8),("ALIGN",(1,1),(-1,-1),"RIGHT"),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.extend([table,Spacer(1,5*mm),P("Model dipilih pada train; kebijakan dipilih pada validation; tabel test merupakan audit setelah kebijakan dikunci. Test referensi ini telah dilihat selama pengembangan dan memerlukan validasi baru.","TinyNote"),PageBreak()])
    story.extend([P("KEPUTUSAN DAN BATAS PENGGUNAAN","Brand"),P("Baca nilai bersama asumsinya","Lead"),P(badge,"Brand"),P(_decision_text(summary))])
    assumptions=_assumption_rows(summary)
    if assumptions:
        t=Table([[P(a,"TableCell"),P(b,"TableCell")] for a,b in assumptions],colWidths=[93*mm,81*mm])
        t.setStyle(TableStyle([("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F1F6F7"),colors.white]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.append(t)
    story.extend([P("Rentang skenario dan risiko","SectionHead"),P(_uncertainty_text(summary)),
        P("Tindakan berikutnya","SectionHead"),P("1. Konfirmasi definisi serta horizon churn, biaya, manfaat kontribusi, dan kapasitas nyata. "
        "2. Validasi pada periode atau populasi yang belum dipakai pengembangan. "
        "3. Uji penawaran terhadap kelompok kontrol sebelum memperluas program."),
        P("Sumber dan keterlacakan","SectionHead"),P(f"Run: {summary['run_id']} | Versi paket proyek: {summary['package_version']}","TinyNote"),
        P(f"Konteks: {summary['data_context']} | Baris sumber: {summary['dataset_rows']:,}","TinyNote"),
        P("SHA-256 sumber: "+summary["source_sha256"][:32]+" "+summary["source_sha256"][32:],"TinyNote"),
        P("Rujukan metode: scikit-learn.org/stable/modules/classification_threshold.html ; "
          "scikit-learn.org/stable/modules/calibration.html . Versi paket, konfigurasi, dan hash split tercatat dalam public_summary.json.","TinyNote")])
    def footer(canvas,document):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#D9E5E8"));canvas.line(18*mm,13*mm,192*mm,13*mm)
        canvas.setFont("Helvetica",7);canvas.setFillColor(colors.HexColor("#5D6B76"))
        canvas.drawString(18*mm,9*mm,"Banking Churn V7 | "+("SYNTHETIC" if summary["synthetic"] else "INTERNAL AUDIT"))
        canvas.drawRightString(192*mm,9*mm,str(document.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return target


def _html(summary,directory):
    e=lambda x:html.escape(str(x))
    d=summary["default_metrics"];m=summary["test_scores"]
    badge="DATA SINTETIS - UJI ALUR, BUKAN HASIL BANK NYATA" if summary["synthetic"] else "DATA REFERENSI - AUDIT INTERNAL"
    cards="".join(f'<div class="card"><span>{e(k)}</span><strong>{e(v)}</strong></div>' for k,v in [
        ("ROC-AUC",f"{m['roc_auc']:.3f}"),("Average precision",f"{m['average_precision']:.3f}"),
        ("Precision default",f"{d['precision']:.1%}"),("Recall default",f"{d['recall']:.1%}")])
    policy_rows="".join("<tr>"+"".join(f"<td>{e(v)}</td>" for v in [r["policy"],int(r["contacts"]),int(r["tp"]),int(r["fp"]),int(r["fn"]),f"{r['precision']:.1%}",f"{r['recall']:.1%}"])+"</tr>" for r in summary["test_policies"])
    params="".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td></tr>" for a,b in _assumption_rows(summary))
    figure=""
    if (directory/"model_audit.png").is_file():
        figure='<img class="audit" alt="ROC dan precision-recall pada test" src="data:image/png;base64,'+base64.b64encode((directory/"model_audit.png").read_bytes()).decode()+'">'
    document=f'''<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banking Churn V7 - Ringkasan</title><style>
:root{{--navy:{NAVY};--teal:{TEAL}}}*{{box-sizing:border-box}}body{{margin:0;background:#edf3f5;color:var(--navy);font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1040px;margin:36px auto;background:#fff;padding:38px 46px;border-radius:16px}}.brand{{color:var(--teal);font-weight:750;letter-spacing:.12em;font-size:13px}}
h1{{font-size:38px;line-height:1.2;margin:12px 0}}h2{{font-size:23px;margin-top:32px}}.badge{{display:inline-block;padding:8px 12px;background:#fff0dc;border:1px solid #e7c38f;border-radius:7px;font-size:13px;font-weight:700}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}}.card{{background:#eaf5f5;padding:17px;border-radius:9px}}.card span{{display:block;font-size:12px}}.card strong{{font-size:29px}}
.audit{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dfe8ec}}th{{background:#eaf1f4}}.table-wrap{{overflow:auto}}.note{{font-size:13px;color:#526571}}code{{overflow-wrap:anywhere}}footer{{margin-top:30px;padding-top:14px;border-top:1px solid #dfe8ec;font-size:12px;color:#526571}}
@media(max-width:650px){{main{{padding:22px;margin:10px}}h1{{font-size:29px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:white}}main{{margin:0;max-width:none;padding:0}}}}
</style><main><div class="brand">BANKING CHURN REDUCTION / V7</div><h1>Ringkasan hasil dan keputusan</h1><p class="badge">{e(badge)}</p>
<p>Model <b>{e(summary['model'])}</b> | Test {m['test_rows']:,} catatan | {m['test_churn']:,} label churn</p><section class="cards">{cards}</section>
<p>Default <b>{e(summary['default_policy'])}</b>, threshold <b>{summary['threshold']:.6f}</b> dari validation: {d['contacts']} prioritas, {d['tp']} churn terdeteksi, {d['fp']} alarm keliru, dan {d['fn']} kasus terlewat. Deteksi label historis belum mengukur pelanggan yang berhasil dipertahankan.</p>
{figure}<h2>Tradeoff kebijakan yang sudah dikunci</h2><div class="table-wrap"><table><thead><tr><th>Kebijakan</th><th>Kontak</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{policy_rows}</tbody></table></div>
<p class="note">Seleksi model pada train; pemilihan threshold pada validation. Test referensi telah dilihat selama pengembangan. Dibutuhkan validasi baru pada periode/populasi lain.</p>
<h2>Keputusan bersyarat dan asumsi</h2><p>{e(_decision_text(summary))}</p><table>{params}</table><h2>Rentang skenario dan risiko</h2><p>{e(_uncertainty_text(summary))}</p>
<h2>Tindakan berikutnya</h2><ol><li>Konfirmasi horizon churn, biaya, manfaat kontribusi, serta kapasitas operasional.</li><li>Validasi pada data baru yang belum dipakai pengembangan.</li><li>Uji penawaran terhadap kelompok kontrol sebelum memperluas program.</li></ol>
<footer>Run <code>{e(summary['run_id'])}</code> | Versi proyek {e(summary['package_version'])}<br>Konteks {e(summary['data_context'])} | {summary['dataset_rows']:,} baris sumber<br>SHA-256 <code>{e(summary['source_sha256'])}</code><br>Parameter distribusi adalah asumsi pengguna. Laporan hanya berisi agregat.<br>Rujukan: <a href="https://scikit-learn.org/stable/modules/classification_threshold.html">threshold</a> dan <a href="https://scikit-learn.org/stable/modules/calibration.html">kalibrasi</a>.</footer></main></html>'''
    path=directory/"executive_summary.html";path.write_text(document,encoding="utf-8")
    return path


def generate_executive_reports(directory):
    directory=Path(directory)
    summary=aggregate_summary(directory)
    public=directory/"public_summary.json"
    public.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if summary["synthetic"] and (directory/"current_report.md").is_file():
        md=(directory/"current_report.md").read_text(encoding="utf-8")
        if not md.startswith("# DATA SINTETIS"):
            (directory/"current_report.md").write_text("# DATA SINTETIS - UJI ALUR, BUKAN HASIL BANK NYATA\n\n"+md,encoding="utf-8")
    outputs={"html":_html(summary,directory),"pdf":_pdf(summary,directory),"summary":public}
    manifest={"run_id":summary["run_id"],"source_sha256":summary["source_sha256"],"data_context":summary["data_context"],
        "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.values()},"customer_records_included":False}
    (directory/"report_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return outputs
