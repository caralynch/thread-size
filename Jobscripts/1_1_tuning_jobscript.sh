#!/bin/bash -l
# ===== SGE resource requests =====
#$ -l h_rt=24:00:00           # wallclock limit (24 hours - adjust based on data size)
#$ -l mem=8G                  # RAM per core (8G x 8 cores = 64G total)
#$ -l tmpfs=400G              # fast local scratch for intermediate files
#$ -pe smp 8                  # 8 CPU cores for parallel processing
#$ -N threadsize_stage1       # job name
#$ -cwd                       # start in submit dir
# #$ -m be                     # (optional) email at begin/end
# #$ -M ucabcpl@ucl.ac.uk 

# ===== Job info =====
echo "===================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "TMPDIR: ${TMPDIR:-/tmp}"
echo "===================================="

if [ -z "${1-}" ]; then
    echo "ERROR: No subreddit supplied."
    echo "Usage: qsub 1_run_thread_size_jobscript.sh <subreddit>"
    exit 1
fi

SUBREDDIT="$1"  # e.g. conspiracy, crypto, politics

# ===== Activate environment =====
source ~/.bashrc
conda activate threadsize 

# ===== Set paths =====
SCRIPT_DIR="/home/ucabcpl/Scratch/thread_size/thread-size/1_Thread_start"
DATA_DIR="/home/ucabcpl/Scratch/thread_size/Outputs/0_preprocessing/${SUBREDDIT}"

# Input files
TRAIN_X="${DATA_DIR}/${SUBREDDIT}_train_X.parquet"
TRAIN_Y="${DATA_DIR}/${SUBREDDIT}_train_Y.parquet"
TEST_X="${DATA_DIR}/${SUBREDDIT}_test_X.parquet"
TEST_Y="${DATA_DIR}/${SUBREDDIT}_test_Y.parquet"

# ===== Verify inputs exist =====
echo "[INFO] Checking required input files..."
for f in "${TRAIN_X}" "${TEST_X}" "${TRAIN_Y}" "${TEST_Y}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# Main output directory
OUTDIR="/home/ucabcpl/Scratch/thread_size/Outputs/1_thread_start/${SUBREDDIT}"
mkdir -p "${OUTDIR}"

LOGDIR="${OUTDIR}/logs"
mkdir -p "${LOGDIR}"

MAX_FEATS=25

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

# ===== Stage 1.2 – feature + weight + threshold tuning =====
TUNING_OUTDIR="${OUTDIR}/2_tuning"
echo "===================================="
echo "[INFO] Stage 1.2 – feature + weight + threshold tuning"
echo "[INFO] Output directory: ${TUNING_OUTDIR}"
echo "===================================="

mkdir -p "${TUNING_OUTDIR}"

run_step "${SUBREDDIT}_tuning" "${SCRIPT_DIR}/2_tuning.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${TUNING_OUTDIR}" \
  --train_X "${TRAIN_X}" \
  --train_y "${TRAIN_Y}" \
  --feats "${MAX_FEATS}"

TUNED_PARAMS="${TUNING_OUTDIR}/tuned_params.jl"
if [ ! -f "${TUNED_PARAMS}" ]; then
    echo "ERROR: Expected tuned params not found: ${TUNED_PARAMS}"
    exit 1
fi
echo "[OK] Found tuned params: ${TUNED_PARAMS}"

echo "===================================="
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "===================================="

exit "${EXIT_CODE}"
