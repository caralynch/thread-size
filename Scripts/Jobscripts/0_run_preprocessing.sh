#!/bin/bash
set -euo pipefail

# Usage and args
if [ $# -lt 4 ]; then
  echo "Usage: $0 <subreddit> <outdir> <comments_file> <threads_file>"
  exit 1
fi

SUBREDDIT=$1
OUTDIR=$2
COMMENTS_IN=$3
THREADS_IN=$4

echo "==== PREPROCESSING ==="
echo "Subreddit:    ${SUBREDDIT}"
echo "Outdir:       ${OUTDIR}"
echo "Comment data: ${COMMENTS_IN}"
echo "Thread data:  ${THREADS_IN}"

# Make dirs
LOGDIR="${OUTDIR}/logs"
TFIDFOUTDIR="${OUTDIR}/tf-idf"
mkdir -p "${LOGDIR}" "${TFIDFOUTDIR}"

SCRIPT_DIR=/home/ucabcpl/Scratch/thread_size/thread-size/Scripts

# Helper: run + log + check
run_step () {
  local name="$1"; shift
  local log="${LOGDIR}/${name}.out"
  echo "== ${name} =="
  echo "Logging to ${log}"
  if ! python -u "$@" > "${log}" 2>&1; then
    echo "${name} failed. See ${log}"
    exit 1
  fi
}

echo "== 1. Constructing features =="
run_step "${SUBREDDIT}_0_1_construct_features" \
  "${SCRIPT_DIR}/0_Preprocessing/1_construct_features.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${OUTDIR}" \
  --comments "${COMMENTS_IN}" \
  --threads "${THREADS_IN}"

# Verify outputs from step 1 exist before continuing
COMMENTS="${OUTDIR}/${SUBREDDIT}_comments_extra_feats.parquet"
THREADS="${OUTDIR}/${SUBREDDIT}_threads_extra_feats.parquet"
if [ ! -f "${COMMENTS}" ] || [ ! -f "${THREADS}" ]; then
  echo "Expected step-1 outputs not found: ${COMMENTS} / ${THREADS}"
  exit 1
fi

echo "Finished constructing features."

echo "== 2. TF-IDF analysis =="
run_step "${SUBREDDIT}_0_2_tfidf_analysis" \
  "${SCRIPT_DIR}/0_Preprocessing/2_tf_idf_analysis.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${TFIDFOUTDIR}" \
  --comments "${COMMENTS}" \
  --threads "${THREADS}" \
  --y-col thread_size \
  --text-col clean_text \
  --train-split 0.8

TRAINDATA="${TFIDFOUTDIR}/${SUBREDDIT}_svd_enriched_train_data.parquet"
TESTDATA="${TFIDFOUTDIR}/${SUBREDDIT}_svd_enriched_test_data.parquet"
if [ ! -f "${TRAINDATA}" ] || [ ! -f "${TESTDATA}" ]; then
  echo "Expected TF-IDF output not found: ${TRAINDATA} / ${TESTDATA}"
  exit 1
fi
echo "Finished TF-IDF analysis."


echo "== 3. Building model-ready dataframes =="

run_step "${SUBREDDIT}_0_3_model_data" \
  "${SCRIPT_DIR}/0_Preprocessing/3_model_data.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${OUTDIR}" \
  --train "${TRAINDATA}" \
  --test "${TESTDATA}" 

# Check final artifacts
X_TRAIN="${OUTDIR}/${SUBREDDIT}_train_X.parquet"
if [ ! -f "${X_TRAIN}" ]; then
  echo "Note: if your 3_model_data.py writes a different filename, update this check."
else
  echo "Found: ${X_TRAIN}"
fi

echo "Finished building model-ready dataframes."

echo "==== Finished preprocessing ===="