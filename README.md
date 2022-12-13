# FLAME GPU 2 Template for Python3

This repository can be used as a template for creating your own FLAME GPU 2 simulations or ensembles using the Python3 interface, using NVRTC, or the experimental native python approach.

For details on how to develop a model using FLAME GPU 2, refer to the [userguide & API documentation](https://docs.flamegpu.com/).

## C++ (CUDA) Interface

FLAME GPU 2 also provides a python-based interface for writing models. If you wish to use this instead of the python2 interface, see [FLAMEGPU/FLAMEGPU2-model-template-cpp](https://github.com/FLAMEGPU/FLAMEGPU2-model-template-cpp).

## Dependencies

+ [Python](https://www.python.org/) `>= 3.7`
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.0` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) `>= 3.5` NVIDIA GPU.
+ pyflamegpu - the python bindings for [FLAME GPU](https://github.com/FLAMEGPU/FLAMEGPU2) `>= 2`

## Getting pyflamegpu

`pyflamegpu` is currently available as pre-built python binary wheels, or can be built from source.

### Pre-compiled pyflamegpu

Pre-built python wheels are available for Windows and Linux, for a range of Python versions on `x86_64` systems.
It is not currently available through any python package repositories.

To install a pre-built version of `pyflamegpu`:

1. Download the appropriate `.whl` for the [Latest Release](https://github.com/FLAMEGPU/FLAMEGPU2/releases/latest)
2. Optionally create and activate a python `venv` or Conda environment
3. Install the wheel locally via pip. See the release notes for details.

### Building pyflamegpu

If the available python wheels are not appropriate for your system, or you wish to build with different CMake configuration options (i.e. `FLAMEGPU_SEATBELTS=OFF` for improved performance with reduced safety checks) you can build your own copy of pyflamegpu.

1. Clone the main [FLAMEGPU/FLAMEGPU2 git repository](https://github.com/FLAMEGPU/FLAMEGPU2) or download an [archived release](https://github.com/FLAMEGPU/FLAMEGPU2/releases).
2. Create a build directory and navigate to it.
3. Configure CMake with `FLAMEGPU_BUILD_PYTHON` set to `ON`.
    + See the main [FLAMEGPU/FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2) repository for further information on CMake configuration options
4. Build the `pyflamegpu` target
5. Optionally create and activate a python `venv` or Conda environment
6. Install the wheel (from `build/lib/<config>/python/dist/`) locally via pip

## Usage

Once `pyflamegpu` is installed into your local python installation or activated virtual environment, you can invoke the example model via:

```bash
python3 model.py <arguments>
```

Use `-h/--help` to see what command line arguments are available for the Simulation or Ensemble within the model

```bash
python3 model.py --help
```

## Documentation and Support

For general information on FLAME GPU, Usage of FLAME GPU `>= 2` and support see:

+ [Website](https://flamegpu.com/)
+ [Documentation and User Guide](https://docs.flamegpu.com)
+ [GitHub Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions)
+ [GitHub Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues)
