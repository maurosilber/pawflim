import hypothesis
import hypothesis.extra.numpy as st_np
import hypothesis.strategies as st
import numpy as np

from .pawflim import _inverse


def invertible(x: np.ndarray) -> bool:
    det = np.linalg.det(x)
    if np.isclose(det, 0) or not np.isfinite(det):
        return False
    else:
        return True


@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=(2, 2),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_infinity=False,
            allow_nan=False,
        ),
    ).filter(invertible)
)
def test_inverse(data):
    assert np.allclose(_inverse(data), np.linalg.inv(data))
