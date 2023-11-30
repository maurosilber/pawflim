# pawFLIM: denoising via adaptive binning for FLIM datasets

![PyPi](https://img.shields.io/pypi/pyversions/pawflim.svg)
[![PyPi](https://img.shields.io/pypi/v/pawflim.svg)](https://pypi.python.org/pypi/pawflim)
[![License](https://img.shields.io/github/license/maurosilber/smo)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/DOI-10.1088%2F2050--6120%2Faa72ab-green)](https://doi.org/10.1088/2050-6120/aa72ab)

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
data[0] = ...  # number of photons
data[1] = ...  # n-th (conjugated) Fourier coefficient
data[2] = ...  # 2n-th (conjugated) Fourier coefficient

denoised = pawflim(data, n_sigmas=2)

phasor = denoised[1] / denoised[0]
```

Note that we use the standard FLIM definition for the $n$-th phasor $r$:

$$ r_n = \\frac{R_n}{R_0} $$

where

$$ R_n = \\int I(t) , e^{i n \\omega t} dt $$

is the $n$-th (conjugated) Fourier coefficient.

See the notebook in
[examples](https://github.com/maurosilber/pawflim/blob/main/examples/simulated_data.ipynb)
for an example with simulated data.
