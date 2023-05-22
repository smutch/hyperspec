# hyperspec

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/hyperspec.svg)](https://pypi.org/project/hyperspec) -->
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperspec.svg)](https://pypi.org/project/hyperspec) -->

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)

## Installation

```console
pip install git+https://github.com/smutch/hyperspec.git
```

## Usage

### CLI

Hyperspec comes with a CLI to run common tasks such as cropping captures and carrying out registration.

The assumption is that the data is in the following directory structure:

```
capture1
├── capture1.png
├── ...
└── results
    ├── ...
    └── REFLECTANCE_capture1.hdr
capture2
├── capture2.png
├── ...
└── results
    ├── ...
    └── REFLECTANCE_capture2.hdr
...
```

To interactively crop all of these captures and store the bounds of the crops in a json file:

```bash
hyperspec crop . bounds.json
```

To register capture1 vs capture2 using the bounds from the previous step:

```bash
hyperspec register capture1/results/REFLECTANCE_capture1.hdr capture2/results/REFLECTANCE_capture2.hdr bounds.json registered-1_2.zarr
```

### Library

See the `examples` directory for a few examples of how to use the library.

## Notes

Much of the documentation for this project was auto-generated using the awesome
[write-the](https://write-the.wytamma.com/) tool (see [#1](https://github.com/smutch/hyperspec/pull/1)).
Go check it out!

## License

`hyperspec` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
