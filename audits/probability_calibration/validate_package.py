#!/usr/bin/env python3
"""Validate the completed aggregate-only audit package."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
REQUIRED=["README.md","report.md","report.html","artifact.json","provenance.md","metric_definitions.md","chart_map.md",
 "audit_probability_calibration.py","build_report_artifact.py","figures/thread_initiation_reliability.png","figures/thread_initiation_reliability.svg",
 "figures/thread_size_reliability.png","figures/thread_size_reliability.svg","outputs/final_model_configurations.csv","outputs/chronological_partitions.csv",
 "outputs/evidence_matrix.csv","outputs/probability_metrics.csv","outputs/reliability_bins.csv",
 "outputs/existing_curve_reconciliation.csv","outputs/existing_curve_summary.csv","outputs/validation_checks.csv",
 "outputs/source_inventory.csv","outputs/run_record.json"]
def digest(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 missing=[x for x in REQUIRED if not (HERE/x).is_file()]; assert not missing,missing
 checks=pd.read_csv(HERE/"outputs/validation_checks.csv"); assert len(checks)==45 and checks.status.eq("PASS").all()
 cfg=pd.read_csv(HERE/"outputs/final_model_configurations.csv"); assert len(cfg)==6 and cfg.configuration_matches_published_selection.all()
 ev=pd.read_csv(HERE/"outputs/evidence_matrix.csv"); assert len(ev)==6
 assert ev.matched_uncalibrated_probabilities.eq("no").all() and ev.before_after_calibration_effect_identifiable.eq("no").all()
 rb=pd.read_csv(HERE/"outputs/reliability_bins.csv")
 groups=rb.groupby(["subreddit","task","class"]); assert groups.size().isin([7,10]).all() and groups.size().loc[("r/CryptoCurrency","four-class thread size")].eq(7).all()
 expected={"r/Conspiracy":2279,"r/CryptoCurrency":2964,"r/politics":13069}
 for (sub,task,cl),g in groups: assert g.n.sum()==expected[sub],(sub,task,cl,g.n.sum())
 rec=json.loads((HERE/"outputs/run_record.json").read_text()); assert rec["checks_failed"]==0 and rec["models_loaded_or_scored"] is False
 art=json.loads((HERE/"artifact.json").read_text()); assert art["surface"]=="report" and art["snapshot"]["status"]=="ready"
 html=(HERE/"report.html").read_text(errors="replace"); assert "Study 1 probability-calibration evidence audit" in html
 rows=[]
 for p in sorted(x for x in HERE.rglob("*") if x.is_file() and x.name!="package_manifest.csv"):
  rows.append({"path":p.relative_to(HERE).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p)})
 pd.DataFrame(rows).to_csv(HERE/"package_manifest.csv",index=False,lineterminator="\n")
 print(json.dumps({"status":"PASS","required_files":len(REQUIRED),"checks":len(checks),"reliability_series":len(groups),"manifest_files":len(rows)},indent=2))
 return 0
if __name__=="__main__": raise SystemExit(main())
