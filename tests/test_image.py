from contextlib import ExitStack as DoesNotRaise

import numpy as np
import pytest

import svlm.utils.image as image


@pytest.mark.parametrize(
    "img, contour, label, mlt, expected_result, exception",
    [
        (
            np.full((500, 500, 3), 255, dtype=np.uint8),
            np.array([]),
            "",
            {"default": 200},
            None,
            pytest.raises(AssertionError),
        ),  # empty contour
        (
            np.full((600, 600, 3), 255, dtype=np.uint8),
            np.array(
                [
                    [[555, 276]],
                    [[554, 277]],
                    [[554, 277]],
                    [[554, 278]],
                    [[554, 279]],
                    [[554, 281]],
                    [[554, 282]],
                    [[554, 284]],
                    [[553, 285]],
                    [[553, 286]],
                    [[552, 287]],
                    [[552, 288]],
                    [[551, 288]],
                    [[551, 306]],
                    [[552, 307]],
                    [[553, 307]],
                    [[559, 300]],
                    [[560, 300]],
                    [[564, 297]],
                    [[565, 297]],
                    [[570, 291]],
                    [[570, 290]],
                    [[571, 289]],
                    [[571, 288]],
                    [[569, 286]],
                    [[569, 284]],
                    [[568, 283]],
                    [[568, 277]],
                    [[566, 276]],
                ]
            ),
            "",
            {"default": 200},
            True,
            DoesNotRaise(),
        ),  # one contour
    ],
)
def test_isbright(
    img: np.ndarray,
    contour: np.ndarray,
    label: str,
    mlt: dict,
    expected_result: bool,
    exception: Exception,
) -> None:
    with exception:
        result = image.isbright(img, contour, label, mlt)
        assert result == expected_result
