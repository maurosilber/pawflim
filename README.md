# pawFLIM: denoising via adaptive binning for FLIM datasets

## Installation

pawFLIM can be installed from PyPI:

```
pip install pawflim
```

## Usage

```python
import numpy as np
from pawflim import pawflim

data = np.empty((3, *shape), dtype=complex)
data[0] = ... # number of photons
data[1] = ... # n-th fourier coefficient
data[2] = ... # 2n-th fourier coefficient

denoised = pawflim(data, n_sigmas=2)

phasor = denoised[1] / denoised[0]
```

See the notebook in
[examples](https://github.com/maurosilber/binlets-paper/blob/main/examples/simulated_data.ipynb)
for an example with simulated data.
