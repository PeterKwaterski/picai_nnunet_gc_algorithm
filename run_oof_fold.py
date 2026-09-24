#!/usr/bin/env python3
"""Run out-of-fold nnU-Net inference for one cross-validation fold."""

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
TASK = "Task2201_picai_baseline"
TRAINER = "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints"
NETWORK = "3d_fullres"
CHECKPOINT = "model_best"

NNUNET_INP_DIR = Path("/opt/algorithm/nnunet/input")
NNUNET_OUT_DIR = Path("/opt/algorithm/nnunet/output")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--gc-cases-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=None,
        help="Optional pooled OOF output directory (one map per case).",
    )
    return parser.parse_args()


def load_subject_list(splits_dir: Path, fold: int) -> list[str]:
    split_file = splits_dir / f"ds-config-valid-fold-{fold}.json"
    with split_file.open() as fp:
        return json.load(fp)["subject_list"]


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


def predict_case(fold: int, results_dir: Path, image_paths: list[Path]) -> tuple[sitk.Image, float]:
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
        TASK,
        "-i",
        str(NNUNET_INP_DIR),
        "-o",
        str(NNUNET_OUT_DIR),
        "-m",
        NETWORK,
        "-tr",
        TRAINER,
        "-f",
        str(fold),
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
    print(f"Fold {args.fold} on {'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'}")

    fold_output_dir = args.output_dir / f"fold_{args.fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    if args.combined_dir is not None:
        args.combined_dir.mkdir(parents=True, exist_ok=True)

    subject_list = load_subject_list(args.splits_dir, args.fold)

    processed = 0
    skipped = 0
    for case_id in subject_list:
        det_map_path = fold_output_dir / f"{case_id}_detection_map.mha"
        if det_map_path.exists():
            skipped += 1
            continue

        case_dir = args.gc_cases_dir / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing GC case directory: {case_dir}")

        image_paths = case_image_paths(case_dir)
        det_map, case_score = predict_case(args.fold, args.results_dir, image_paths)
        atomic_image_write(det_map, det_map_path)

        score_path = fold_output_dir / f"{case_id}_case_level_likelihood.json"
        score_path.write_text(json.dumps(case_score) + "\n")

        if args.combined_dir is not None:
            combined_path = args.combined_dir / f"{case_id}_detection_map.mha"
            atomic_image_write(det_map, combined_path)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(subject_list)} cases (skipped {skipped})")

    print(
        f"Done fold {args.fold}: processed={processed}, skipped={skipped}, "
        f"total={len(subject_list)}"
    )


if __name__ == "__main__":
    main()
