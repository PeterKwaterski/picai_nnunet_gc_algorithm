import json
import os
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import UniqueImagesValidator, UniquePathIndicesValidator
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample

from inference_v2 import extract_lesion_candidates_cropped, probability_to_reference

DATASET = "220"
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainerPICAI_FLCE"
CHECKPOINT = "checkpoint_best.pth"
PLANS = "nnUNetResEncUNetMPlans"


class csPCaAlgorithmV2(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators={
                "input_image": (UniqueImagesValidator(), UniquePathIndicesValidator()),
            }
        )
        self.image_input_dirs = [
            Path("/input/images/transverse-t2-prostate-mri"),
            Path("/input/images/transverse-adc-prostate-mri"),
            Path("/input/images/transverse-hbv-prostate-mri"),
        ]
        self.scan_paths = [self._single_scan(folder) for folder in self.image_input_dirs]
        self.input_dir = Path("/opt/algorithm/nnunet/input")
        self.output_dir = Path("/opt/algorithm/nnunet/output")
        self.results_dir = Path("/opt/algorithm/nnUNet_results")
        self.detection_map_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.confidence_path = Path("/output/cspca-case-level-likelihood.json")
        for folder in (self.input_dir, self.output_dir, self.detection_map_path.parent):
            folder.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _single_scan(folder: Path) -> Path:
        scans = list(folder.glob("*.mha"))
        if len(scans) != 1:
            raise RuntimeError(f"Expected one scan in {folder}, found {len(scans)}")
        return scans[0]

    def preprocess_input(self) -> sitk.Image:
        sample = Sample(scans=[sitk.ReadImage(str(path)) for path in self.scan_paths])
        sample.preprocess()
        for index, scan in enumerate(sample.scans):
            atomic_image_write(scan, self.input_dir / f"scan_{index:04d}.nii.gz")
        return sample.scans[0]

    def predict(self) -> np.ndarray:
        os.environ["nnUNet_results"] = str(self.results_dir)
        subprocess.check_call(
            [
                "nnUNetv2_predict",
                "-i",
                str(self.input_dir),
                "-o",
                str(self.output_dir),
                "-d",
                DATASET,
                "-c",
                CONFIGURATION,
                "-tr",
                TRAINER,
                "-p",
                PLANS,
                "-chk",
                CHECKPOINT,
                "-f",
                "0",
                "1",
                "2",
                "3",
                "--save_probabilities",
                "-npp",
                "2",
                "-nps",
                "1",
            ]
        )
        probabilities = np.load(self.output_dir / "scan.npz")["probabilities"]
        if probabilities.shape[0] != 2:
            raise RuntimeError(f"Expected binary probabilities, got shape {probabilities.shape}")
        return probabilities[1].astype(np.float32, copy=False)

    def process(self):
        preprocessed_t2 = self.preprocess_input()
        probability = self.predict()
        original_t2 = sitk.ReadImage(str(self.scan_paths[0]))
        probability_image = probability_to_reference(probability, preprocessed_t2, original_t2)

        detection = extract_lesion_candidates_cropped(sitk.GetArrayFromImage(probability_image))
        detection_image = sitk.GetImageFromArray(detection)
        detection_image.CopyInformation(original_t2)
        atomic_image_write(detection_image, self.detection_map_path)

        with self.confidence_path.open("w") as fp:
            json.dump(float(np.max(detection)), fp)


if __name__ == "__main__":
    csPCaAlgorithmV2().process()
