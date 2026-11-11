#!/usr/bin/env python3
"""Run four-fold ensemble inference on a locked holdout subject list."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample

from process import convert_to_original_extent, extract_lesion_candidates_cropped

IMAGE_DIRS = [
    "images/transverse-t2-prostate-mri",
    "images/transverse-adc-prostate-mri",
    "images/transverse-hbv-prostate-mri",
]
DEFAULT_TASK = "Task2201_picai_baseline"
TRAINER = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints"
NETWORK = "3d_fullres"
CHECKPOINT = "model_best"

NNUNET_INP_DIR = Path("/opt/algorithm/nnunet/input")
NNUNET_OUT_DIR = Path("/opt/algorithm/nnunet/output")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-list",
        type=Path,
        required=True,
        help="JSON file with subject_list (e.g. holdout_subject_list.json).",
    )
    parser.add_argument("--gc-cases-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ensemble-folds",
        type=str,
        default="0,1,2,3",
        help="Comma-separated nnU-Net fold indices to ensemble (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=os.environ.get("NNUNET_TASK", DEFAULT_TASK),
    )
    return parser.parse_args()


def case_image_paths(case_dir: Path) -> list[Path]:
    paths = []
    for rel_dir in IMAGE_DIRS:
        matches = sorted((case_dir / rel_dir).glob("*.mha"))
        if not matches:
            raise FileNotFoundError(f"No .mha files in {case_dir / rel_dir}")
        paths.append(matches[0])
    return paths


def clear_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def predict_case_ensemble(
    task: str,
    results_dir: Path,
    image_paths: list[Path],
    ensemble_folds: str,
) -> tuple[sitk.Image, float]:
    os.environ["RESULTS_FOLDER"] = str(results_dir)

    clear_dir(NNUNET_INP_DIR)
    clear_dir(NNUNET_OUT_DIR)

    sample = Sample(scans=[sitk.ReadImage(str(path)) for path in image_paths])
    sample.preprocess()
    for i, scan in enumerate(sample.scans):
        atomic_image_write(scan, NNUNET_INP_DIR / f"scan_{i:04d}.nii.gz")

    cmd = [
        "nnUNet_predict",
        "-t",
        task,
        "-i",
        str(NNUNET_INP_DIR),
        "-o",
        str(NNUNET_OUT_DIR),
        "-m",
        NETWORK,
        "-tr",
        TRAINER,
        "-f",
        ensemble_folds,
        "-chk",
        CHECKPOINT,
        "--save_npz",
        "--num_threads_preprocessing",
        "2",
        "--num_threads_nifti_save",
        "1",
    ]
    subprocess.check_call(cmd)

    pred_path = NNUNET_OUT_DIR / "scan.npz"
    pred = np.array(np.load(pred_path)["softmax"][1]).astype("float32")
    pred_path.unlink()

    convert_to_original_extent(
        pred=pred,
        pkl_path=NNUNET_OUT_DIR / "scan.pkl",
        dst_path=NNUNET_OUT_DIR / "softmax.nii.gz",
    )

    pred_ensemble = sitk.ReadImage(str(NNUNET_OUT_DIR / "softmax.nii.gz"))
    detection_map = extract_lesion_candidates_cropped(
        pred=sitk.GetArrayFromImage(pred_ensemble),
        threshold="dynamic",
    )

    reference_scan = sitk.ReadImage(str(image_paths[0]))
    det_map = sitk.GetImageFromArray(detection_map)
    det_map.CopyInformation(reference_scan)
    return det_map, float(np.max(detection_map))


def main():
    args = parse_args()
    with args.subject_list.open() as fp:
        subject_list = json.load(fp)["subject_list"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for case_id in subject_list:
        det_map_path = args.output_dir / f"{case_id}_detection_map.mha"
        if det_map_path.exists():
            skipped += 1
            continue

        case_dir = args.gc_cases_dir / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing GC case directory: {case_dir}")

        image_paths = case_image_paths(case_dir)
        det_map, case_score = predict_case_ensemble(
            args.task,
            args.results_dir,
            image_paths,
            args.ensemble_folds,
        )
        atomic_image_write(det_map, det_map_path)

        score_path = args.output_dir / f"{case_id}_case_level_likelihood.json"
        score_path.write_text(json.dumps(case_score) + "\n")

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(subject_list)} (skipped {skipped})")

    print(
        f"Holdout inference complete: processed={processed}, skipped={skipped}, "
        f"total={len(subject_list)}"
    )


if __name__ == "__main__":
    main()
