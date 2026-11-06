import pytest

np = pytest.importorskip("numpy")
sitk = pytest.importorskip("SimpleITK")

from inference_v2 import probability_to_reference


def test_probability_to_reference_matches_reference_geometry():
    source = sitk.Image([4, 5, 6], sitk.sitkFloat32)
    source.SetSpacing((2.0, 2.0, 2.0))
    reference = sitk.Image([8, 10, 12], sitk.sitkFloat32)
    reference.SetSpacing((1.0, 1.0, 1.0))
    reference.SetOrigin((3.0, 4.0, 5.0))
    probability = np.ones((6, 5, 4), dtype=np.float32)

    actual = probability_to_reference(probability, source, reference)

    assert actual.GetSize() == reference.GetSize()
    assert actual.GetSpacing() == reference.GetSpacing()
    assert actual.GetOrigin() == reference.GetOrigin()
    assert actual.GetDirection() == reference.GetDirection()
