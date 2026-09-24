#!/bin/bash
#SBATCH --job-name=picai-nnunet-oof
#SBATCH --partition=dgx
#SBATCH --chdir=/home/ad.msoe.edu/kwaterskip/Research/picai/baselines/picai_nnunet_gc_algorithm
#SBATCH --output=logs/oof_%A_%a.log
#SBATCH --error=logs/oof_%A_%a.err
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPTPATH="$(pwd)"
PROJECT_ROOT="$(cd "${SCRIPTPATH}/../.." && pwd)"

GC_CASES_DIR="${GC_CASES_DIR:-${PROJECT_ROOT}/data/gc_cases}"
SPLITS_DIR="${SPLITS_DIR:-${SCRIPTPATH}/data/splits/picai_nnunet}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPTPATH}/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPTPATH}/results/oof}"
COMBINED_DIR="${COMBINED_DIR:-${OUTPUT_DIR}/combined}"
SIF="${SIF:-${SCRIPTPATH}/picai_baseline_nnunet_processor.sif}"

FOLD="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID via --array}"

mkdir -p "${SCRIPTPATH}/logs" "${OUTPUT_DIR}" "${COMBINED_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-$$}-${FOLD}"
mkdir -p "${MPLCONFIGDIR}"

NNUNET_SCRATCH="/tmp/nnunet-${SLURM_JOB_ID:-$$}-${FOLD}"
RESULTS_SCRATCH="/tmp/results-${SLURM_JOB_ID:-$$}-${FOLD}"
mkdir -p "${NNUNET_SCRATCH}/input" "${NNUNET_SCRATCH}/output" "${RESULTS_SCRATCH}"

export MKL_THREADING_LAYER=GNU

echo "Project root:  ${PROJECT_ROOT}"
echo "Fold:          ${FOLD}"
echo "GC cases:      ${GC_CASES_DIR}"
echo "Splits:        ${SPLITS_DIR}"
echo "Results:       ${RESULTS_DIR}"
echo "Results scratch: ${RESULTS_SCRATCH}"
echo "Output:        ${OUTPUT_DIR}/fold_${FOLD}"
echo "Combined:      ${COMBINED_DIR}"

module purge
module load singularity/3.10.0
module load cuda/12.9

# nnUNet_predict writes training logs into the results folder during init
cp -a "${RESULTS_DIR}/." "${RESULTS_SCRATCH}/"

singularity exec --nv --no-home \
  --env MKL_THREADING_LAYER=GNU \
  --bind "${GC_CASES_DIR}:/gc_cases:ro" \
  --bind "${SPLITS_DIR}:/splits:ro" \
  --bind "${RESULTS_SCRATCH}:/opt/algorithm/results" \
  --bind "${NNUNET_SCRATCH}:/opt/algorithm/nnunet" \
  --bind "${OUTPUT_DIR}:/output" \
  --bind "${COMBINED_DIR}:/combined" \
  --bind "${SCRIPTPATH}/run_oof_fold.py:/opt/algorithm/run_oof_fold.py:ro" \
  "${SIF}" \
  python3 /opt/algorithm/run_oof_fold.py \
    --fold "${FOLD}" \
    --gc-cases-dir /gc_cases \
    --splits-dir /splits \
    --results-dir /opt/algorithm/results \
    --output-dir /output \
    --combined-dir /combined

echo "Fold ${FOLD} complete."
