# cmml-inflam

[![Multiple OS](https://github.com/I-Bouros/cmml-inflam/actions/workflows/os-unittests.yml/badge.svg)](https://github.com/I-Bouros/cmml-inflam/actions/workflows/os-unittests.yml)
[![Multiple OS](https://github.com/I-Bouros/cmml-inflam/actions/workflows/os-unittests.yml/badge.svg)](https://github.com/I-Bouros/cmml-inflam/actions/workflows/os-unittests.yml)
[![Copyright License](https://github.com/I-Bouros/cmml-inflam/actions/workflows/check-copyright.yml/badge.svg)](https://github.com/I-Bouros/cmml-inflam/actions/workflows/check-copyright.yml)
[![Documentation Status](https://readthedocs.org/projects/cmml-inflam/badge/?version=latest)](https://cmml-inflam.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/I-Bouros/cmml-inflam/branch/main/graph/badge.svg?token=D2K0BR7OgN)](https://codecov.io/gh/I-Bouros/cmml-inflam)
[![Style tests (flake8)](https://github.com/I-Bouros/cmml-inflam/actions/workflows/flake8-style-test.yml/badge.svg)](https://github.com/I-Bouros/cmml-inflam/actions/workflows/flake8-style-test.yml)

A collection of stochastic process models used in the modelling the evolution of CMML.

All features of our software are described in detail in our
[full API documentation](https://cmml-inflam.readthedocs.io/en/latest/).

More details on CMML modeling and inflammation can be found in these
papers:

## References

## Installation procedure
***
One way to install the module is to download the repositiory to your machine of choice and type the following commands in the terminal. 
```bash
git clone https://github.com/I-Bouros/cmml-inflam.git
cd ../path/to/the/file
```

A different method to install this is using `pip`:

```bash
pip install -e .
```

## Usage

```python
import cmmlinflam

# create and run a Gillespie routine for a STEM cell population for given time frame
algo = cmmlinflam.StemGillespie()
algo.simulate(parameters, start_time, end_time)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)