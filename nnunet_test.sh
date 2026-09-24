#!/bin/bash
#SBATCH --job-name=picai-nnunet
#SBATCH --partition=dgx
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
module load singularity/3.10.0
module load cuda/12.9


CASE_INPUT="test"
CASE_OUTPUT="outputs"
SIF="picai_baseline_nnunet_processor.sif"

mkdir -p "${CASE_OUTPUT}/images/cspca-detection-map"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"

NNUNET_SCRATCH="/tmp/nnunet-${SLURM_JOB_ID:-$$}"
mkdir -p "${NNUNET_SCRATCH}"
RESULTS_SCRATCH="/tmp/results-${SLURM_JOB_ID:-$$}"
mkdir -p "${RESULTS_SCRATCH}"

export MKL_THREADING_LAYER=GNU

singularity exec --no-home \
  --bind "${RESULTS_SCRATCH}:/results_scratch" \
  "${SIF}" \
  cp -a /opt/algorithm/results/. /results_scratch/

singularity exec --nv --no-home \
  --env MKL_THREADING_LAYER=GNU \
  --bind "${CASE_INPUT}:/input" \
  --bind "${CASE_OUTPUT}:/output" \
  --bind "${NNUNET_SCRATCH}:/opt/algorithm/nnunet" \
  --bind "${RESULTS_SCRATCH}:/opt/algorithm/results" \
  "${SIF}" \
  python3 /opt/algorithm/process.py