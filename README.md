# FLAME GPU 2 Python Template Example

This repository acts as an example to be used as a template for creating standalone FLAME GPU 2 models which use the python interface.

For details on how to develop a model using FLAME GPU 2, refer to the [user guide & api documentation](https://docs.flamegpu.com/).

## Dependencies

+ [Python](https://www.python.org/) `>= 3.6` for python integration
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.0` and a Compute Capability `>= 3.5` NVIDIA GPU.
+ pyflamegpu - the python bindings for [FLAME GPU](https://github.com/FLAMEGPU/FLAMEGPU2) `>= 2`

## Getting pyflamegpu

### Pre-compiled pyflamegpu

Pre-built python wheels are available for Windows and Linux, for a range of Python versions on x86_64 systems.

To install a pre-built version of `pyflamegpu`:

1. Download the appropriate `.whl` for the [Latest Release](https://github.com/FLAMEGPU/FLAMEGPU2/releases/latest)
2. Optionally create and activate a python `venv` or Conda environment
3. Install the wheel locally via pip

### Building pyflamegpu

If the available python wheels are not appropriate for your system, or you wish to build with different CMake configuration options (i.e. `SEATBELTS=OFF` for improved performance with reduced safety checks) you can build your own copy of pyflamegpu.

1. Clone the main [FLAMEGPU/FLAMEGPU2  git repository](https://github.com/FLAMEGPU/FLAMEGPU2) or download an [archived release](https://github.com/FLAMEGPU/FLAMEGPU2/releases).
2. Create a build directory and navigate to it.
3. Configure CMake with `BUILD_SWIG_PYTHON` set to `ON`.
    + See the main [FLAMEGPU/FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2) repository for further information on CMake configuration options
4. Build the `pyflamegpu` target
5. Optionally create and activate a python `venv` or Conda environment
6. Install the wheel (from `build/lib/<config>/python/dist/`) locally via pip

## Usage

Once `pyflamegpu` is installed into your local python installation or activated virtual environment, you can invoke the example model via:

```bash
python3 model.py <arguments>
```

Use `-h/--help` to see what command line arguments are available.

```bash
python3 model.py --help
```

## Documentation and Support

For general information on FLAME GPU, Usage of FLAME GPU `>= 2` and support see:

+ [Website](https://flamegpu.com/)
+ [Documentation and User Guide](https://docs.flamegpu.com)
+ [GitHub Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions)
+ [GitHub Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues)

