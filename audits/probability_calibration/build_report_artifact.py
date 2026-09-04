#!/usr/bin/env python3
"""Build the canonical portable report manifest from reviewed audit outputs."""
from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent; OUT=HERE/"outputs"
def rows(name):
 with (OUT/name).open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def source(i,label,path,desc):
 q={"engine":"duckdb","language":"sql","sql":f"SELECT * FROM read_csv_auto('{path}', header=true)",
    "description":desc,"tables_used":[path],"filters":["Aggregate audit outputs only; no row-level probabilities or identifiers"]} if path.endswith(".csv") else None
 x={"id":i,"label":label,"path":path}
 if q:x["query"]=q
 return x
def table(i,title,dataset,sid,columns,sort):
 return {"id":i,"title":title,"dataset":dataset,"sourceId":sid,"defaultSort":{"field":sort,"direction":"asc"},
  "columns":[{"field":f,"label":l,"type":t} for f,l,t in columns]}

def main():
 cfg=rows("final_model_configurations.csv"); ev=rows("evidence_matrix.csv"); met=rows("probability_metrics.csv")
 bins=rows("reliability_bins.csv"); curves=rows("existing_curve_summary.csv"); checks=rows("validation_checks.csv")
 overall=[r for r in met if r["scope"]=="overall"]
 binary=[{"subreddit":r["subreddit"],"bin":int(r["bin"]),"mean_predicted_probability":float(r["mean_predicted_probability"]),
          "observed_fraction":float(r["observed_fraction"]),"n":int(r["n"])} for r in bins if r["task"]=="thread initiation"]
 configurations=[{"subreddit":r["subreddit"],"task":r["task"],"predictors":int(r["candidate_predictor_count"]),
  "features":r["features"],"calibration":r["calibration_method"],"labels_or_ranges":r["target_labels"] if r["task"]=="thread initiation" else r["thread_size_class_ranges"]} for r in cfg]
 evidence=[{"subreddit":r["subreddit"],"task":r["task"],"calibrated_rows_labels":r["chronological_holdout_probabilities"].startswith("yes") and r["chronological_holdout_true_labels"].startswith("yes"),
  "executed_test_curve_inputs":r["executed_holdout_curve_inputs"],"surviving_test_plot":r["surviving_holdout_curve_plot"],
  "matched_uncalibrated":r["matched_uncalibrated_probabilities"],"calibration_effect_identifiable":r["before_after_calibration_effect_identifiable"]} for r in ev]
 metrics=[{"subreddit":r["subreddit"],"task":r["task"],"metric":r["metric"],"value":round(float(r["value"]),6),"definition":r["definition"]} for r in overall]
 curve_rows=[{"subreddit":r["subreddit"],"surviving_plot":r["existing_plot_split"],"rule":r["curve_rule"],
  "test_bins":int(r["test_realized_nonempty_bins"]),"test_n":int(r["test_denominator"]),"oof_bins":int(r["oof_realized_nonempty_bins"]),"oof_n":int(r["oof_denominator"])} for r in curves]
 validation=[{"subreddit":r["subreddit"],"task":r["task"],"check":r["check"],"status":r["status"],"detail":r["detail"]} for r in checks]
 sources=[
  source("report_text","Detailed audit report","report.md","Narrative findings, limitations and thesis-safe wording."),
  source("config_source","Final model configurations","outputs/final_model_configurations.csv","Selected compact configuration reconciliation."),
  source("evidence_source","Artefact survival matrix","outputs/evidence_matrix.csv","Matched probability, label, curve and uncalibrated evidence status."),
  source("metrics_source","Held-out probability metrics","outputs/probability_metrics.csv","Metrics calculated from saved calibrated probabilities and labels."),
  source("bins_source","Held-out reliability bins","outputs/reliability_bins.csv","Ten-quantile-bin reliability calculations from saved calibrated rows."),
  source("curves_source","Existing curve provenance","outputs/existing_curve_summary.csv","Executed binary curve binning, denominators and plot attribution."),
  source("checks_source","Validation checks","outputs/validation_checks.csv","Automated configuration, row, label and probability reconciliation checks."),
  source("inventory_source","Checksummed source inventory","outputs/source_inventory.csv","Paths, roles, sizes, timestamps and SHA-256 hashes for source artefacts."),
  source("definitions","Metric definitions","metric_definitions.md","Exact definitions and interpretive limits.")]
 tables=[
  table("config_table","Validated final compact configurations","configurations","config_source",[("subreddit","Subreddit","text"),("task","Task","text"),("predictors","Predictors","number"),("features","Features","text"),("calibration","Calibration","text"),("labels_or_ranges","Labels / ranges","text")],"subreddit"),
  table("evidence_table","What survives for each selected model","evidence","evidence_source",[("subreddit","Subreddit","text"),("task","Task","text"),("calibrated_rows_labels","Calibrated rows + labels","boolean"),("executed_test_curve_inputs","Executed test curve inputs","text"),("surviving_test_plot","Surviving test plot","text"),("matched_uncalibrated","Matched uncalibrated","text"),("calibration_effect_identifiable","Effect identifiable","text")],"subreddit"),
  table("metrics_table","Chronological-holdout probabilistic metrics","metrics","metrics_source",[("subreddit","Subreddit","text"),("task","Task","text"),("metric","Metric","text"),("value","Value","number"),("definition","Definition","text")],"subreddit"),
  table("curves_table","Existing binary curve attribution and denominators","curve_rows","curves_source",[("subreddit","Subreddit","text"),("surviving_plot","Surviving plot","text"),("rule","Rule","text"),("test_bins","Test bins","number"),("test_n","Test N","number"),("oof_bins","OOF bins","number"),("oof_n","OOF N","number")],"subreddit"),
  table("validation_table","Validation checks","validation","checks_source",[("subreddit","Subreddit","text"),("task","Task","text"),("check","Check","text"),("status","Status","text"),("detail","Detail","text")],"subreddit")]
 chart={"id":"binary_reliability_chart","title":"Binary held-out reliability by subreddit","subtitle":"Saved calibrated Started probabilities; ten quantile bins per subreddit.","type":"line","dataset":"binary_reliability","sourceId":"bins_source",
  "encodings":{"x":{"field":"mean_predicted_probability","type":"quantitative","label":"Mean predicted probability"},
   "y":{"field":"observed_fraction","type":"quantitative","label":"Observed fraction"},
   "color":{"field":"subreddit","type":"nominal","label":"Subreddit"}},"xAxisTitle":"Mean predicted probability","yAxisTitle":"Observed fraction","legend":{"show":True},"layout":"full"}
 blocks=[
  {"id":"title","type":"markdown","body":"# Study 1 probability-calibration evidence audit"},
  {"id":"answer","type":"markdown","sourceId":"evidence_source","body":"## Technical summary\n\nCalibrated row-level probabilities and true labels survive for all six selected chronological holdouts. Matched uncalibrated probabilities survive for none, so calibration's effect is not identifiable. The existing binary PNGs show OOF training-population curves because they overwrite the test plots; no four-class reliability curve was generated by the original pipeline."},
  {"id":"config_heading","type":"markdown","body":"## All six artefact sets reconcile to the published compact selections\n\nThe selected directories, predictor counts, features, labels and class ranges agree across publication selectors, workbooks, saved feature matrices and prediction files."},
  {"id":"config","type":"table","tableId":"config_table","layout":"full"},
  {"id":"evidence_heading","type":"markdown","body":"## Probability evidence survives; matched raw probabilities do not\n\nThe distinction is model-wide: empirical assessment of the retained calibrated probabilities is possible, but a before/after calibration comparison is not."},
  {"id":"evidence","type":"table","tableId":"evidence_table","layout":"full"},
  {"id":"visual_heading","type":"markdown","sourceId":"bins_source","body":"## Held-out reliability is measurable from retained rows\n\nThe chart shows the binary Started probability. The audit package provides separate print-oriented PNG/SVG figures for thread initiation and four-class thread size; the latter shows all four one-vs-rest classes."},
  {"id":"visual","type":"chart","chartId":"binary_reliability_chart","layout":"full"},
  {"id":"metrics_heading","type":"markdown","body":"## Probabilistic scores are descriptive, not causal calibration effects\n\nBrier score and log loss combine calibration with other properties of probabilistic predictions. ECE/MCE use a requested ten-quantile-bin rule; duplicate edges can reduce the realised non-empty count."},
  {"id":"metrics","type":"table","tableId":"metrics_table","layout":"full"},
  {"id":"curves_heading","type":"markdown","body":"## The surviving binary plots are OOF, not chronological test\n\nBoth curve coordinate sets survive, but the code writes both plots to the same filename. OOF observations are fold-held-out within the training population."},
  {"id":"curves","type":"table","tableId":"curves_table","layout":"full"},
  {"id":"methods","type":"markdown","sourceId":"report_text","body":"## Methodology\n\nThe audit used only saved artefacts. It matched split-local indices to preprocessing order and retained thread IDs; checked chronological separation, target derivations, configurations, probability ranges/simplexes, and threshold/argmax predictions; reproduced binary test curve coordinates; and calculated transparent held-out summaries. No model object was loaded or scored."},
  {"id":"limits","type":"markdown","sourceId":"report_text","body":"## Limitations and robustness\n\nNo matched uncalibrated output survives. Stage 2 reliability diagrams are audit reconstructions rather than original pipeline plots. Bin-based gaps depend on the binning rule. This audit establishes internal artefact consistency and provenance, not external generalisability."},
  {"id":"recommendation","type":"markdown","sourceId":"report_text","body":"## Chapter 4 recommendation\n\nAdd one compact appendix table and the two task-specific held-out figures, with a short Chapter 4 cross-reference. State calibration procedures and held-out diagnostics, but do not claim calibration improved the models. Do not reuse the existing binary PNGs as final-test plots. Thesis-safe Methods, appendix and viva wording is provided in `report.md`."},
  {"id":"validation_heading","type":"markdown","body":"## Validation\n\nAll 45 automated reconciliation checks pass. Full checksums and calculation metadata remain in the package."},
  {"id":"validation","type":"table","tableId":"validation_table","layout":"full"},
  {"id":"further","type":"markdown","sourceId":"report_text","body":"## Further questions\n\nA future authorised reproducibility run should retain matched raw and calibrated probabilities, give test and OOF plots distinct filenames, and pre-specify multiclass calibration summaries."}]
 now=datetime.now(timezone.utc).isoformat()
 manifest={"version":1,"surface":"report","title":"Study 1 probability-calibration evidence audit",
  "description":"Provenance-first audit of calibration evidence for six validated compact Study 1 models.","generatedAt":now,
  "cards":[],"charts":[chart],"tables":tables,"sources":sources,"blocks":blocks}
 artifact={"surface":"report","manifest":manifest,"snapshot":{"version":1,"generatedAt":now,"status":"ready",
  "datasets":{"configurations":configurations,"evidence":evidence,"metrics":metrics,"curve_rows":curve_rows,
              "validation":validation,"binary_reliability":binary}},"sources":sources}
 (HERE/"artifact.json").write_text(json.dumps(artifact,indent=2)+"\n",encoding="utf-8")
 print(HERE/"artifact.json")
 return 0
if __name__=="__main__": raise SystemExit(main())
