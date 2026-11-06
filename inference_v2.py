import numpy as np
import SimpleITK as sitk
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates


def probability_to_reference(probability: np.ndarray, source: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a probability volume from nnU-Net input geometry onto the original T2 grid."""
    image = sitk.GetImageFromArray(probability.astype(np.float32, copy=False))
    image.CopyInformation(source)
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def extract_lesion_candidates_cropped(prediction: np.ndarray) -> np.ndarray:
    original_size = prediction.shape
    prediction = crop_or_pad(prediction, (20, 384, 384))
    prediction = crop_or_pad(prediction, original_size)
    return extract_lesion_candidates(prediction, threshold="dynamic")[0]
