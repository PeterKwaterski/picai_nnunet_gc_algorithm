import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample

from inference_v2 import extract_lesion_candidates_cropped, probability_to_reference

IMAGE_DIRS = (
    "images/transverse-t2-prostate-mri",
    "images/transverse-adc-prostate-mri",
    "images/transverse-hbv-prostate-mri",
)
NNUNET_INPUT = Path("/opt/algorithm/nnunet/input")
NNUNET_OUTPUT = Path("/opt/algorithm/nnunet/output")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--gc-cases-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-dir", type=Path)
    return parser.parse_args()


def clear_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def image_paths(case_dir: Path) -> list[Path]:
    paths = []
    for relative in IMAGE_DIRS:
        matches = sorted((case_dir / relative).glob("*.mha"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one scan in {case_dir / relative}, found {len(matches)}")
        paths.append(matches[0])
    return paths


def predict_case(folds: list[str], results_dir: Path, paths: list[Path]) -> tuple[sitk.Image, float]:
    clear_dir(NNUNET_INPUT)
    clear_dir(NNUNET_OUTPUT)
    sample = Sample(scans=[sitk.ReadImage(str(path)) for path in paths])
    sample.preprocess()
    for index, scan in enumerate(sample.scans):
        atomic_image_write(scan, NNUNET_INPUT / f"scan_{index:04d}.nii.gz")

    os.environ["nnUNet_results"] = str(results_dir)
    subprocess.check_call(
        [
            "nnUNetv2_predict",
            "-i", str(NNUNET_INPUT),
            "-o", str(NNUNET_OUTPUT),
            "-d", "220",
            "-c", "3d_fullres",
            "-tr", "nnUNetTrainerPICAI_FLCE",
            "-p", "nnUNetResEncUNetMPlans",
            "-f", *folds,
            "-chk", "checkpoint_best.pth",
            "--save_probabilities",
            "-npp", "2",
            "-nps", "1",
        ]
    )

    probability = np.load(NNUNET_OUTPUT / "scan.npz")["probabilities"][1]
    reference = sitk.ReadImage(str(paths[0]))
    probability_image = probability_to_reference(probability, sample.scans[0], reference)
    detection = extract_lesion_candidates_cropped(sitk.GetArrayFromImage(probability_image))
    detection_image = sitk.GetImageFromArray(detection)
    detection_image.CopyInformation(reference)
    return detection_image, float(np.max(detection))


def main():
    args = parse_args()
    split_file = args.splits_dir / f"ds-config-valid-fold-{args.fold}.json"
    subject_list = json.loads(split_file.read_text())["subject_list"]
    fold_output = args.output_dir / f"fold_{args.fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    if args.combined_dir:
        args.combined_dir.mkdir(parents=True, exist_ok=True)

    for case_id in subject_list:
        output = fold_output / f"{case_id}_detection_map.mha"
        if output.exists():
            continue
        detection, score = predict_case([str(args.fold)], args.results_dir, image_paths(args.gc_cases_dir / case_id))
        atomic_image_write(detection, output)
        (fold_output / f"{case_id}_case_level_likelihood.json").write_text(json.dumps(score) + "\n")
        if args.combined_dir:
            atomic_image_write(detection, args.combined_dir / output.name)


if __name__ == "__main__":
    main()
