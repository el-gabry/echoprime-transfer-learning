import numpy as np

from echoprime_transfer.preprocessing import crop_and_scale


def test_crop_and_scale_shape():
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    output = crop_and_scale(frame)

    assert output.shape == (224, 224, 3)