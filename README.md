# AthenaK Dust Tests

This repository collects validation and numerical-test material for the Lagrangian dust module developed in [athenak-dust/athenak](https://github.com/athenak-dust/athenak).

It includes:

- Jupyter notebooks documenting test setups, results, and analysis;
- AthenaK input files used by the tests; and
- small stand-alone Python concept tests; and
- plots and other supporting test material when useful.

The notebooks cover basic dust–gas drag validation, comparisons with published test problems, shearing-box and streaming-instability tests, and CPU/GPU runs. Each notebook contains its own build, input, and execution notes.

For reproducibility, clone this repository alongside the AthenaK source repository so that the directories are arranged as:

```text
athenak/
athenak_dust_tests/
```

The code under test is maintained in the `dust` branch of [athenak-dust/athenak](https://github.com/athenak-dust/athenak/tree/dust).
