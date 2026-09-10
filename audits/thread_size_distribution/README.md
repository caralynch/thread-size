# Study 1 observed thread-size distribution audit

This package independently audits the raw observed thread-size distributions for
the exact final Study 1 populations. It reads the final preprocessing outcomes,
the retained split-to-thread mappings, raw thread/comment parquet files, selected
thread-size evaluation workbooks, pipeline code, and prior validation records.
It does not load or score a model and writes only inside this directory.

Run the full audit with the repository environment:

```bash
/home/cara/anaconda3/envs/2stagemodel/bin/python audit_thread_size_distribution.py --force
```

Re-render the figure from the frozen aggregate CSV inputs only:

```bash
/home/cara/anaconda3/envs/2stagemodel/bin/python render_ccdf.py
```

Primary outputs are `report.md`, `provenance.md`, the CSV files under
`outputs/`, and the SVG/PNG figure under `figures/`.
