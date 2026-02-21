# Building Psi4 (psi4nv) from Source

Tested on Ubuntu 24.04 (x86_64) with GCC 14, CUDA 13, and Conda 24.

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

## 1. Create the Conda Build Environment

Create (or update) the `p4dev` environment from the included recipe, then pin
Python to 3.12 (Python 3.14 is incompatible with Pydantic v1 used by
qcelemental):

```bash
conda env create -n p4dev --file devtools/conda-envs/linux-64-buildrun.yaml
conda install -n p4dev -c conda-forge python=3.12 scipy --yes
```

If the environment already exists, use `conda env update` instead of `create`:

```bash
conda env update -n p4dev --file devtools/conda-envs/linux-64-buildrun.yaml
conda install -n p4dev -c conda-forge python=3.12 scipy --yes
```

This installs all required build and runtime dependencies: MKL, Eigen, Boost
headers, Libint2, Libxc, gau2grid, pybind11, qcelemental, qcengine,
qcmanybody, optking, NumPy, SciPy, and the conda C/C++ compilers.

## 2. cuEST Setup (Optional)

cuEST is disabled by default (`ENABLE_cuEST=OFF`). If you plan to enable it
later, place the cuEST distribution **outside** the Psi4 source tree:

```
/path/to/cuest/
├── include/
│   ├── cuest.h
│   └── h/
├── lib/
│   ├── 12/   (CUDA 12 libraries)
│   └── 13/   (CUDA 13 libraries)
└── python_bindings/
```

The Psi4 source tree contains the CMake find wrapper at
`external/upstream/cuest/CMakeLists.txt`. Point to the distribution at
configure time with `-DcuEST_ROOT=/path/to/cuest`.

## 3. Configure

```bash
conda run -n p4dev cmake -S. -Bobjdir \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(conda run -n p4dev python -c 'import sys; print(sys.prefix)')" \
  -DPython_EXECUTABLE="$(conda run -n p4dev which python)" \
  -G Ninja
```

To enable cuEST, add:

```
  -DENABLE_cuEST=ON \
  -DcuEST_ROOT=/path/to/cuest
```

### Key configure options

| Flag | Default | Description |
|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` or `Debug` |
| `CMAKE_INSTALL_PREFIX` | `/usr/local/psi4` | Install location |
| `ENABLE_cuEST` | `OFF` | Enable cuEST GPU module |
| `cuEST_ROOT` | — | Path to cuEST distribution |
| `ENABLE_OPENMP` | `ON` | OpenMP parallelization |
| `MAX_AM_ERI` | `5` | Max angular momentum for integrals |

See the top of `CMakeLists.txt` for the full list of options.

## 4. Build

```bash
conda run -n p4dev cmake --build objdir -j$(nproc)
```

Typical build time: ~4 minutes on 32 cores.

## 5. Install

```bash
conda run -n p4dev cmake --install objdir
```

## 6. Verify

Run from **outside** the source tree (to avoid the source `psi4/` directory
shadowing the installed package):

```bash
cd /tmp
conda run -n p4dev env PYTHONPATH=/path/to/install/lib/ \
  python -c "import psi4; print(psi4.__version__)"
```

Expected output: `1.11a1.dev19` (or similar).

## 7. Run Tests

```bash
cd objdir
conda run -n p4dev ctest -j$(nproc) --output-on-failure
```

## Troubleshooting

### Python 3.14 import errors

If you see `pydantic.v1.errors.ConfigError: unable to infer type for attribute "doi"`,
your Python version is too new. Pin to 3.12:

```bash
conda install -n p4dev -c conda-forge python=3.12 scipy --yes
```

Then clean and rebuild (steps 3-5).

### Libint2 Boost warning

The warning about `BoostConfig.cmake` path no longer being valid is harmless
and can be ignored. Libint2 still links correctly.

### Source tree shadows installed package

Always run `import psi4` from outside the repo root, or set `PYTHONPATH`
to point to the install directory and ensure the current directory isn't the
source tree.

### Clean rebuild

```bash
rm -rf objdir install
```

Then repeat from step 3.
