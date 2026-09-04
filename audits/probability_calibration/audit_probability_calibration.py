#!/usr/bin/env python3
"""Audit saved calibration evidence; never load or score a model."""
from __future__ import annotations

import hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
OUT, FIG = AUDIT / "outputs", AUDIT / "figures"
SUBS = ("conspiracy", "crypto", "politics")
NAME = {"conspiracy":"r/Conspiracy", "crypto":"r/CryptoCurrency", "politics":"r/politics"}
S1 = {"conspiracy":4, "crypto":4, "politics":4}
S2 = {"conspiracy":3, "crypto":2, "politics":3}
F1 = {
 "conspiracy":["author_freq","question_ratio","domain_freq","subject_length"],
 "crypto":["author_freq","domain_freq","subject_length","question_ratio"],
 "politics":["domain_pagerank","author_freq","domain_freq","hour"],
}
F2 = {
 "conspiracy":["author_freq","domain_freq","question_ratio"],
 "crypto":["author_freq","domain_freq"],
 "politics":["domain_pagerank","domain_freq","author_freq"],
}
RANGES = {
 "conspiracy":["1","2-6","7-19",">=20"],
 "crypto":["1","2-10","11-32",">=33"],
 "politics":["1","2-9","10-30",">=31"],
}
CUTS = {"conspiracy":[1,6,19], "crypto":[1,10,32], "politics":[1,9,30]}
CN = {0:"Stalled",1:"Small",2:"Medium",3:"Large"}

def rel(p): return p.resolve().relative_to(ROOT.resolve()).as_posix()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def save(name, rows):
 assert rows, name
 pd.DataFrame(rows).to_csv(OUT/name,index=False,lineterminator="\n")
def value(df,n,key):
 x=df[(df.n_feats==n)&(df.Key==key)]
 assert len(x)==1,(n,key,len(x))
 return x.iloc[0].Value
def qlist(x): return re.findall(r"['\"]([^'\"]+)['\"]",str(x))
def inv(p,role,sub="all",task="shared"):
 s=p.stat()
 return {"subreddit":NAME.get(sub,sub),"task":task,"role":role,"path":rel(p),"bytes":s.st_size,
  "modified_utc":datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),"sha256":sha(p)}
def check(a,sub,task,name,ok,detail):
 a.append({"subreddit":NAME.get(sub,sub),"task":task,"check":name,"status":"PASS" if ok else "FAIL","detail":detail})
def classes(size,cuts):
 x=np.asarray(size)
 return np.select([x<=cuts[0],x<=cuts[1],x<=cuts[2]],[0,1,2],default=3).astype(int)
def reliability(y,p,k=10):
 """Sklearn quantile allocation plus counts, ECE and MCE."""
 y,p=np.asarray(y,dtype=int),np.asarray(p,dtype=float)
 edges=np.unique(np.percentile(p,np.linspace(0,100,k+1)))
 ids=np.searchsorted(edges[1:-1],p)
 rows=[]
 for i in range(len(edges)-1):
  m=ids==i
  if not m.any(): continue
  mp,obs=float(p[m].mean()),float(y[m].mean())
  rows.append({"bin":i+1,"n":int(m.sum()),"probability_min":float(p[m].min()),
   "probability_max":float(p[m].max()),"mean_predicted_probability":mp,
   "observed_fraction":obs,"absolute_gap":abs(obs-mp)})
 return rows,sum(r["n"]*r["absolute_gap"] for r in rows)/len(y),max(r["absolute_gap"] for r in rows)

def main():
 OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
 cfg=[]; chrono=[]; ev=[]; met=[]; rb=[]; curves=[]; curve_sum=[]; checks=[]; inventory=[]; pdata={}
 shared=[
  (ROOT/"Publication_Outputs/1_Thread_Start/s1_mods.txt","published compact selection"),
  (ROOT/"Publication_Outputs/2_Thread_Size/s2_mods.txt","published compact selection"),
  (ROOT/"Scripts/1_Thread_start/4_run_tuned_model.py","executed Stage 1 evaluation code"),
  (ROOT/"Scripts/2_Thread_size/4_run_tuned_model.py","executed Stage 2 evaluation code"),
  (ROOT/"Scripts/0_Preprocessing/2_tf_idf_analysis.py","chronological split code"),
  (ROOT/"Scripts/0_Preprocessing/3_model_data.py","model-data construction code")]
 inventory += [inv(p,r) for p,r in shared if p.exists()]
 for sub in SUBS:
  yt=ROOT/f"Outputs/0_preprocessing/{sub}/{sub}_train_Y.parquet"
  yv=ROOT/f"Outputs/0_preprocessing/{sub}/{sub}_test_Y.parquet"
  et=ROOT/f"Outputs/0_preprocessing/{sub}/tf-idf/{sub}_svd_enriched_train_data.parquet"
  evp=ROOT/f"Outputs/0_preprocessing/{sub}/tf-idf/{sub}_svd_enriched_test_data.parquet"
  inventory += [inv(p,r,sub,"shared population") for p,r in [(yt,"training outcomes"),(yv,"chronological holdout outcomes"),(et,"training thread identifiers"),(evp,"held-out thread identifiers")]]
  train=pd.read_parquet(yt); test=pd.read_parquet(yv)
  tid=pd.read_parquet(et,columns=["thread_id","timestamp","thread_size","success"])
  vid=pd.read_parquet(evp,columns=["thread_id","timestamp","thread_size","success"])
  ordered=bool(train.timestamp.is_monotonic_increasing and test.timestamp.is_monotonic_increasing)
  separated=bool(train.timestamp.max()<test.timestamp.min())
  ids_ok=bool(train.index.equals(tid.index) and test.index.equals(vid.index) and
   train.timestamp.equals(tid.timestamp) and test.timestamp.equals(vid.timestamp) and
   train.thread_size.equals(tid.thread_size) and test.thread_size.equals(vid.thread_size))
  chrono.append({"subreddit":NAME[sub],"train_n":len(train),"train_start":train.timestamp.min().isoformat(),
   "train_end":train.timestamp.max().isoformat(),"test_n":len(test),"test_start":test.timestamp.min().isoformat(),
   "test_end":test.timestamp.max().isoformat(),"within_split_monotonic":ordered,
   "train_ends_before_test_starts":separated,"split_local_index_train":f"0..{len(train)-1}",
   "split_local_index_test":f"0..{len(test)-1}","thread_id_mapping_survives":ids_ok})
  check(checks,sub,"shared population","chronological partition",ordered and separated,f"train end {train.timestamp.max()}; test start {test.timestamp.min()}")
  check(checks,sub,"shared population","split-local index maps to thread_id",ids_ok,f"{len(train)} train; {len(test)} test")

  # Binary thread initiation.
  n=S1[sub]; run=ROOT/f"Outputs/1_thread_start/{sub}/4_model"; model=run/f"model_{n}"; book=run/"evaluation.xlsx"
  pars=pd.read_excel(book,sheet_name="all_params"); pars["n_feats"]=pars["Unnamed: 0"].ffill().astype(int)
  feats=qlist(value(pars,n,"features")); threshold=float(value(pars,n,"model_threshold")); tuning_threshold=float(value(pars,n,"final_threshold")); cm=feats==F1[sub]
  cfg.append({"subreddit":NAME[sub],"task":"thread initiation","candidate_predictor_count":n,
   "selected_model_directory":rel(model),"features":" | ".join(feats),
   "target_labels":"0=Stalled (thread_size=1) | 1=Started (thread_size>1)",
   "calibration_method":"isotonic","decision_threshold":threshold,"fold_aggregated_tuning_threshold":tuning_threshold,"thread_size_class_ranges":"not applicable",
   "configuration_matches_published_selection":cm})
  files=[(book,"evaluation workbook"),(model/"test_data_results.xlsx","per-model evaluation workbook"),
   (model/"test_started_threads.parquet","held-out row probabilities"),(model/"train_started_threads.parquet","OOF training-row probabilities"),
   (model/"model_data/test_calibration_curve_inputs.jl","executed holdout curve coordinates"),
   (model/"model_data/oof_calibration_curve_inputs.jl","executed OOF curve coordinates"),
   (model/"plots/calibration_curve.png","surviving reliability plot")]
  inventory += [inv(p,r,sub,"thread initiation") for p,r in files]
  rows=pd.read_parquet(model/"test_started_threads.parquet"); y=(test.thread_size.to_numpy()>1).astype(int); p=rows.proba.to_numpy(float)
  idx=np.array_equal(rows["index"].to_numpy(),np.arange(len(y)))
  target=np.array_equal(y,vid.success.to_numpy(int)); pred=np.array_equal(rows.predicted.to_numpy(int),(p>=threshold).astype(int))
  valid=bool(np.isfinite(p).all() and ((0<=p)&(p<=1)).all())
  check(checks,sub,"thread initiation","probability rows match holdout",idx and len(p)==len(y),f"N={len(y)}; split-local ordinal identifiers")
  check(checks,sub,"thread initiation","binary target derivation",target,"success equals 1(thread_size>1)")
  check(checks,sub,"thread initiation","probability validity",valid,f"range [{p.min():.6g}, {p.max():.6g}]")
  check(checks,sub,"thread initiation","saved threshold predictions",pred,f"threshold={threshold:.4f}")
  check(checks,sub,"thread initiation","compact configuration",cm," | ".join(feats))
  br,ece,mce=reliability(y,p); st,sp=calibration_curve(y,p,n_bins=10,strategy="quantile")
  saved_test=joblib.load(model/"model_data/test_calibration_curve_inputs.jl")
  saved_oof=joblib.load(model/"model_data/oof_calibration_curve_inputs.jl")
  same=bool(np.allclose(st,saved_test["prob_true"],rtol=0,atol=1e-12) and np.allclose(sp,saved_test["prob_pred"],rtol=0,atol=1e-12))
  check(checks,sub,"thread initiation","saved test curve inputs reproduce",same,f"{len(st)} non-empty quantile bins")
  for split,saved in [("chronological test",saved_test),("OOF training",saved_oof)]:
   curves += [{"subreddit":NAME[sub],"task":"thread initiation","split":split,"bin":i,
    "saved_observed_fraction":float(a),"saved_mean_predicted_probability":float(b),
    "recomputed_from_rows":split=="chronological test"} for i,(a,b) in enumerate(zip(saved["prob_true"],saved["prob_pred"]),1)]
  tm=(model/"model_data/test_calibration_curve_inputs.jl").stat().st_mtime
  om=(model/"model_data/oof_calibration_curve_inputs.jl").stat().st_mtime
  pm=(model/"plots/calibration_curve.png").stat().st_mtime; oof_plot=tm<om<=pm
  curve_sum.append({"subreddit":NAME[sub],"task":"thread initiation",
   "existing_plot_split":"OOF training (overwrote test plot)" if oof_plot else "timestamp sequence inconclusive",
   "existing_plot_is_chronological_holdout":False if oof_plot else "unresolved",
   "curve_rule":"sklearn calibration_curve; n_bins=10; strategy=quantile","test_requested_bins":10,
   "test_realized_nonempty_bins":len(saved_test["prob_true"]),"test_denominator":len(y),"oof_requested_bins":10,
   "oof_realized_nonempty_bins":len(saved_oof["prob_true"]),"oof_denominator":len(train),
   "attribution_basis":"source loop writes test then OOF to same PNG; mtimes follow that order"})
  rb += [{"subreddit":NAME[sub],"task":"thread initiation","class":"Started","split":"chronological test","binning_rule":"10 quantile bins",**r} for r in br]
  pdata[(sub,"thread initiation")]=[{"class":"Started",**r} for r in br]
  met += [
   {"subreddit":NAME[sub],"task":"thread initiation","split":"chronological test","scope":"overall","metric":"Brier score","value":brier_score_loss(y,p),"definition":"mean((p-y)^2); lower is better"},
   {"subreddit":NAME[sub],"task":"thread initiation","split":"chronological test","scope":"overall","metric":"log loss","value":log_loss(y,p,labels=[0,1]),"definition":"mean negative log probability of true label"},
   {"subreddit":NAME[sub],"task":"thread initiation","split":"chronological test","scope":"overall","metric":"AUC","value":roc_auc_score(y,p),"definition":"ROC area; discrimination, not calibration"},
   {"subreddit":NAME[sub],"task":"thread initiation","split":"chronological test","scope":"overall","metric":"ECE","value":ece,"definition":"count-weighted absolute gap; 10 quantile bins"},
   {"subreddit":NAME[sub],"task":"thread initiation","split":"chronological test","scope":"overall","metric":"MCE","value":mce,"definition":"maximum absolute gap; 10 quantile bins"}]
  ev.append({"subreddit":NAME[sub],"task":"thread initiation","chronological_holdout_probabilities":"yes; calibrated row-level proba",
   "chronological_holdout_true_labels":"yes; derivable and mapped to thread_id","executed_holdout_curve_inputs":"yes",
   "surviving_holdout_curve_plot":"no; overwritten by OOF plot","existing_plot_population":"OOF fold-held-out training rows",
   "matched_uncalibrated_probabilities":"no","before_after_calibration_effect_identifiable":"no",
   "empirical_calibration_assessment":"possible on chronological holdout from retained calibrated rows"})

  # Four-class thread size.
  n=S2[sub]; run=ROOT/f"Outputs/2_thread_size/{sub}/4_model"; model=run/f"model_{n}"; book=run/"evaluation.xlsx"
  pars=pd.read_excel(book,sheet_name="model_params"); feats=qlist(value(pars,n,"features")); bintext=str(value(pars,n,"bins")); cm=feats==F2[sub]
  cfg.append({"subreddit":NAME[sub],"task":"four-class thread size","candidate_predictor_count":n,
   "selected_model_directory":rel(model),"features":" | ".join(feats),"target_labels":"0=Stalled | 1=Small | 2=Medium | 3=Large",
   "calibration_method":"sigmoid (one-vs-rest CalibratedClassifierCV)","decision_threshold":"argmax",
   "thread_size_class_ranges":" | ".join(f"{CN[i]}={v}" for i,v in enumerate(RANGES[sub])),
   "configuration_matches_published_selection":cm})
  files=[(book,"evaluation workbook"),(model/"test_data_results.xlsx","per-model evaluation workbook"),
   (model/"test_preds.csv","held-out labels and probabilities"),(model/"oof_preds.csv","OOF labels and probabilities"),
   (model/"model_data/y_proba.parquet","held-out probability matrix"),(model/"model_data/y_test.parquet","held-out encoded labels"),
   (model/"model_data/y_pred.parquet","held-out predicted labels"),(model/"model_data/oof_proba.parquet","OOF probability matrix"),
   (model/"model_data/y_train.parquet","training encoded labels"),(model/"model_data/X_test.parquet","selected held-out predictors")]
  inventory += [inv(p,r,sub,"four-class thread size") for p,r in files]
  csv=pd.read_csv(model/"test_preds.csv"); pf=pd.read_parquet(model/"model_data/y_proba.parquet"); p=pf.to_numpy(float)
  y=pd.read_parquet(model/"model_data/y_test.parquet").iloc[:,0].to_numpy(int)
  yp=pd.read_parquet(model/"model_data/y_pred.parquet").iloc[:,0].to_numpy(int); x=pd.read_parquet(model/"model_data/X_test.parquet")
  derived=classes(test.thread_size,CUTS[sub]); idx=np.array_equal(csv["index"],np.arange(len(y)))
  labels=np.array_equal(y,derived) and np.array_equal(csv.true_class.to_numpy(int),y)
  probs=np.allclose(csv[[f"proba_class_{i}" for i in range(4)]].to_numpy(),p,rtol=0,atol=1e-15)
  preds=np.array_equal(yp,p.argmax(1)) and np.array_equal(csv.predicted_class.to_numpy(int),yp)
  simplex=bool(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-12))
  check(checks,sub,"four-class thread size","row identifiers",idx and len(y)==len(test),f"N={len(y)}; split-local ordinal identifiers")
  check(checks,sub,"four-class thread size","encoded targets match class ranges",labels," | ".join(RANGES[sub]))
  check(checks,sub,"four-class thread size","CSV and Parquet probabilities match",probs,"four columns, row-aligned")
  check(checks,sub,"four-class thread size","probability simplex",simplex,f"max |sum(p)-1|={np.max(np.abs(p.sum(1)-1)):.3g}")
  check(checks,sub,"four-class thread size","saved predictions equal argmax",preds,"CSV and Parquet agree")
  check(checks,sub,"four-class thread size","compact configuration",cm and list(x.columns)==feats," | ".join(feats))
  check(checks,sub,"four-class thread size","workbook bin definition recovered","0.692147" in bintext,bintext[:180])
  oh=np.eye(4)[y]; total=float(np.mean(np.sum((p-oh)**2,1)))
  met += [
   {"subreddit":NAME[sub],"task":"four-class thread size","split":"chronological test","scope":"overall","metric":"multiclass Brier score","value":total,"definition":"mean row-wise sum of four squared errors; range 0-2"},
   {"subreddit":NAME[sub],"task":"four-class thread size","split":"chronological test","scope":"overall","metric":"normalized multiclass Brier","value":total/4,"definition":"multiclass Brier divided by 4; mean per class"},
   {"subreddit":NAME[sub],"task":"four-class thread size","split":"chronological test","scope":"overall","metric":"log loss","value":log_loss(y,p,labels=[0,1,2,3]),"definition":"mean negative log probability of true class"}]
  pr=[]
  for k in range(4):
   yy=(y==k).astype(int); br,ece,mce=reliability(yy,p[:,k]); label=f"{CN[k]} ({RANGES[sub][k]})"
   for r in br:
    rb.append({"subreddit":NAME[sub],"task":"four-class thread size","class":label,"split":"chronological test","binning_rule":"10 quantile bins, one-vs-rest",**r}); pr.append({"class":label,**r})
   met += [
    {"subreddit":NAME[sub],"task":"four-class thread size","split":"chronological test","scope":f"class {k}: {CN[k]}","metric":"one-vs-rest Brier score","value":brier_score_loss(yy,p[:,k]),"definition":"mean((p_k-I[y=k])^2)"},
    {"subreddit":NAME[sub],"task":"four-class thread size","split":"chronological test","scope":f"class {k}: {CN[k]}","metric":"ECE","value":ece,"definition":"count-weighted absolute gap; 10 quantile bins"},
    {"subreddit":NAME[sub],"task":"four-class thread size","split":"chronological test","scope":f"class {k}: {CN[k]}","metric":"MCE","value":mce,"definition":"maximum absolute gap; 10 quantile bins"}]
  pdata[(sub,"four-class thread size")]=pr
  ev.append({"subreddit":NAME[sub],"task":"four-class thread size","chronological_holdout_probabilities":"yes; calibrated four-class rows",
   "chronological_holdout_true_labels":"yes; encoded labels and thread_id mapping","executed_holdout_curve_inputs":"no",
   "surviving_holdout_curve_plot":"no","existing_plot_population":"none; no reliability curve was generated",
   "matched_uncalibrated_probabilities":"no","before_after_calibration_effect_identifiable":"no",
   "empirical_calibration_assessment":"possible retrospectively on chronological holdout from retained calibrated rows"})

 # Separate appendix figures, derived only from saved probabilities/labels.
 plt.rcParams.update({"font.size":11,"axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10})
 colors=["#0072B2","#E69F00","#009E73","#CC79A7"]
 styles=[("o","-"),("s","--"),("^","-."),("D",":")]
 # Thread-initiation figure: one panel per subreddit.
 fig,axes=plt.subplots(1,3,figsize=(11.4,4.15),sharex=True,sharey=True,constrained_layout=True)
 for i,sub in enumerate(SUBS):
  ax=axes[i]; ax.plot([0,1],[0,1],"--",color="#333333",lw=1.3,label="Ideal")
  grouped=pd.DataFrame(pdata[(sub,"thread initiation")]).groupby("class",sort=False)
  for color,(marker,linestyle),(label,g) in zip(colors,styles,grouped):
   ax.plot(g.mean_predicted_probability,g.observed_fraction,marker=marker,linestyle=linestyle,
    ms=5.2,lw=1.8,color=color,label=label,markerfacecolor="white",markeredgewidth=1.2)
  n_holdout=chrono[i]["test_n"]; ax.set_title(f"{NAME[sub]}\nChronological holdout N={n_holdout:,}",pad=9)
  ax.set(xlim=(0,1),ylim=(0,1)); ax.grid(False)
  ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
  ax.legend(fontsize=8.5,loc="upper left",frameon=False,handlelength=2.4)
 fig.supxlabel("Mean predicted probability",fontsize=11.5)
 fig.supylabel("Observed fraction",fontsize=11.5)
 fig.savefig(FIG/"thread_initiation_reliability.png",dpi=300,bbox_inches="tight")
 fig.savefig(FIG/"thread_initiation_reliability.svg",bbox_inches="tight")
 plt.close(fig)

 # Thread-size figure: outcome-class rows by subreddit columns.
 fig,axes=plt.subplots(4,3,figsize=(11.4,11.8),sharex=True,sharey=True,constrained_layout=True)
 for j,sub in enumerate(SUBS):
  n_holdout=chrono[j]["test_n"]; axes[0,j].set_title(f"{NAME[sub]}\nN={n_holdout:,}",pad=10,fontsize=13,fontweight="semibold")
  frame=pd.DataFrame(pdata[(sub,"four-class thread size")])
  for k in range(4):
   ax=axes[k,j]; label=f"{CN[k]} ({RANGES[sub][k]})"; g=frame[frame["class"]==label]
   ax.plot([0,1],[0,1],"--",color="#333333",lw=1.2)
   ax.plot(g.mean_predicted_probability,g.observed_fraction,marker="o",linestyle="-",
    ms=5.0,lw=1.8,color="#0072B2",markerfacecolor="white",markeredgewidth=1.2)
   ax.text(0.04,0.94,label,transform=ax.transAxes,ha="left",va="top",fontsize=10.5,fontweight="normal")
   ax.set(xlim=(0,1),ylim=(0,1)); ax.grid(False)
   ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
 fig.supxlabel("Mean predicted probability",fontsize=11.5)
 fig.supylabel("Observed fraction",fontsize=11.5)
 fig.savefig(FIG/"thread_size_reliability.png",dpi=300,bbox_inches="tight")
 fig.savefig(FIG/"thread_size_reliability.svg",bbox_inches="tight")
 plt.close(fig)
 for name,rows in [("final_model_configurations.csv",cfg),("chronological_partitions.csv",chrono),("evidence_matrix.csv",ev),
  ("probability_metrics.csv",met),("reliability_bins.csv",rb),("existing_curve_reconciliation.csv",curves),
  ("existing_curve_summary.csv",curve_sum),("validation_checks.csv",checks),("source_inventory.csv",inventory)]: save(name,rows)
 failures=[x for x in checks if x["status"]!="PASS"]
 commit=subprocess.run(["git","-C",str(ROOT/"Scripts"),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 record={"generated_utc":datetime.now(timezone.utc).isoformat(),"script":rel(Path(__file__)),"python":sys.version,
  "library_versions":{"pandas":pd.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__},
  "scripts_git_commit":commit,"models_loaded_or_scored":False,"existing_pipeline_files_modified":False,
  "scope":"three subreddits x two selected compact Study 1 tasks","checks_total":len(checks),"checks_failed":len(failures),"failed_checks":failures}
 (OUT/"run_record.json").write_text(json.dumps(record,indent=2)+"\n")
 print(json.dumps({"status":"ok" if not failures else "failed",**record},indent=2))
 return 0 if not failures else 1

if __name__=="__main__": raise SystemExit(main())
