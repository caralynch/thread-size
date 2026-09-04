# Probability-calibration evidence audit

This package audits the surviving probability-calibration evidence for the six validated compact Study 1 models. It is read-only with respect to the modelling pipeline: no estimator was loaded, fitted, tuned, recalibrated, or scored. Only already-saved probabilities, labels, curve inputs, workbooks, logs, source code, and preprocessing rows were read.

Primary deliverables:

- `report.html`: self-contained reader report.
- `report.md`: concise text report and thesis-safe wording.
- `figures/thread_initiation_reliability.png` and `figures/thread_size_reliability.png`: appendix-ready task-specific reconstructions from saved chronological holdout probabilities and labels.
- `outputs/evidence_matrix.csv`: survival/gap assessment for each model.
- `outputs/source_inventory.csv`: checksummed provenance inventory.
- `outputs/validation_checks.csv`: 45 reconciliation checks.

Reproduce the calculations with the project environment:

```bash
/home/cara/anaconda3/envs/2stagemodel/bin/python audit_probability_calibration.py
```

The script writes only inside this directory. `run_record.json` records versions, the `Scripts` Git commit, and the explicit assertion that no model was loaded or scored.

