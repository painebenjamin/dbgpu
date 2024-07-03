<div align="center">
<img src="https://github.com/painebenjamin/dbgpu/assets/57536852/a632cd1e-337d-4819-aa3f-4d13e1259fe4" />
</div>
<hr />
<p align="center">
A small, easy-to-use open source database of over 2000 GPUs with architecture, manufacturing, API support and performance details sourced from <a href="https://www.techpowerup.com/gpu-specs/" target="_blank">TechPowerUp</a>.
</p>
<p align="center">
    <img src="https://img.shields.io/static/v1?label=painebenjamin&message=dbgpu&logo=github&color=111111" alt="painebenjamin - dbgpu">
    <img src="https://img.shields.io/github/stars/painebenjamin/dbgpu?style=social" alt="stars - dbgpu">
    <img src="https://img.shields.io/github/forks/painebenjamin/dbgpu?style=social" alt="forks - dbgpu"><br />
    <a href="https://github.com/painebenjamin/dbgpu/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-111111" alt="License"></a>
    <a href="https://github.com/painebenjamin/dbgpu/releases/"><img src="https://img.shields.io/github/tag/painebenjamin/dbgpu?include_prereleases=&sort=semver&color=111111" alt="GitHub tag"></a>
    <a href="https://github.com/painebenjamin/dbgpu/releases/"><img alt="GitHub release (with filter)" src="https://img.shields.io/github/v/release/painebenjamin/dbgpu?color=111111"></a>
    <a href="https://pypi.org/project/dbgpu"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/dbgpu?color=111111"></a>
    <a href="https://github.com/painebenjamin/dbgpu/releases/"><img alt="GitHub all releases" src="https://img.shields.io/github/downloads/painebenjamin/dbgpu/total?logo=github&color=111111"></a>
    <a href="https://pypi.org/project/dbgpu"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/dbgpu?logo=python&logoColor=white&color=111111"></a>
</p>

# Installation

DBGPU is available on PyPI and can be installed with pip:

```sh
pip install dbgpu
```

In order to be as minimal as possible (the package is only `170kb` compressed,) some features are only available as additional dependencies. To install any additional package, use `pip install dbgpu[package]`:
- `dbgpu[tabulate]` will install [tabulate](https://github.com/astanin/python-tabulate/) for pretty-printing tables.
- `dbgpu[fuzz]` will install [thefuzz](https://github.com/seatgeek/thefuzz) for fuzzy searching.
- `dbgpu[build]` will install [requests](https://docs.python-requests.org/en/master/), [beautifulsoup4](https://beautiful-soup-4.readthedocs.io/) and [tqdm](https://tqdm.github.io/) for building the database.
- `dbgpu[socks]` will install [PySocks](https://github.com/Anorov/PySocks) for SOCKS proxy support.
- `dbgpu[all]` will install all optional dependencies.

# Usage
## Python API

```py
from dbgpu import GPUDatabase

database = GPUDatabase.default()
spec = database["GeForce GTX 1080"]
# Using fuzzy search (slower):
# spec = database.search("GTX 1080")
print(spec)
```

This is the output without `tabulate` available; see below for an example with it installed.

```sh
----------------
GeForce GTX 1080
----------------
GPU Name: GP104
Manufacturer: NVIDIA
Architecture: Pascal
Foundry: TSMC
Process Size: 16 nm
Transistor Count: 7.2 billion
Transistor Density: 22.9 million/mm²
Die Size: 314 mm²
Chip Package: BGA-2150
Release Date: 2016-05-27
Generation: GeForce 10
Bus Interface: PCIe 3.0 x16
Base Clock: 1,607 MHz
Boost Clock: 1,733 MHz
Memory Clock: 1,251 MHz
Memory Size: 8.0 GB
Memory Type: GDDR5X
Memory Bus: 256 bit
Memory Bandwidth: 320.3 GB/s
Shading Units: 2,560
Texture Mapping Units: 160
Render Output Processors: 64
Streaming Multiprocessors: 20
Tensor Cores: Unknown
Ray Tracing Cores: Unknown
L1 Cache: 48.0 KB
L2 Cache: 2.0 MB
Thermal Design Power: 180 W
Board Length: 267 mm
Board Width: 112 mm
Board Slot Width: Dual-slot
Suggested PSU: 450 W
Power Connectors: 1x 8-pin
Display Connectors: 1x DVI, 1x HDMI 2.0, 3x DisplayPort 1.4a
DirectX Version: 12.1
OpenGL Version: 4.6
Vulkan Version: 1.3
OpenCL Version: 3.0
CUDA Version: 6.1
Shader Model Version: 6.7
Pixel Rate: 110.9 GPixel/s
Texture Rate: 277.3 GTexel/s
Half Float Performance: 138.6 GFLOP/s
Single Float Performance: 8.9 TFLOP/s
Double Float Performance: 277.3 GFLOP/s
```

Available fields and their types are:

```py
class GPUSpecification:
    manufacturer: MANUFACTURER_LITERAL # Manufacturer of the GPU
    name: str  # Common name of the GPU
    gpu_name: str  # Name of the GPU as per the manufacturer
    generation: str  # GPU generation, e.g. "GeForce 30"
    base_clock_mhz: Optional[float]  # Base clock speed in MHz
    boost_clock_mhz: Optional[float]  # Boost clock speed in MHz
    architecture: Optional[str]  # Architecture of the GPU, e.g. "Ampere"
    foundry: Optional[str]  # Foundry where the GPU was manufactured
    process_size_nm: Optional[int]  # Process size in whole nanometers
    transistor_count_m: Optional[float]  # Number of transistors in the GPU (in millions)
    transistor_density_k_mm2: Optional[float]  # Transistor density in thousands of transistors per square millimeter
    die_size_mm2: Optional[float]  # Die size in square millimeters
    chip_package: Optional[str]  # Package of the GPU chip
    release_date: Optional[date]  # Release date of the GPU
    bus_interface: Optional[str]  # Bus interface of the GPU
    memory_clock_mhz: Optional[float]  # Memory clock speed in MHz
    memory_size_gb: Optional[float]  # Size of the GPU memory in GB
    memory_bus_bits: Optional[int]  # Memory bus width in bits
    memory_bandwidth_gb_s: Optional[float]  # Memory bandwidth in GB/s
    memory_type: Optional[str]  # Type of the GPU memory, e.g. "GDDR6"
    shading_units: int  # Number of shading units in the GPU
    texture_mapping_units: int  # Number of texture mapping units (TMUs) in the GPU
    render_output_processors: int  # Number of render output processors (ROPs) in the GPU
    streaming_multiprocessors: int  # Number of streaming multiprocessors (SMs) in the GPU
    tensor_cores: int  # Number of tensor cores in the GPU
    ray_tracing_cores: int  # Number of ray tracing cores in the GPU
    l1_cache_kb: float  # L1 cache size in KB
    l2_cache_mb: float  # L2 cache size in MB
    thermal_design_power_w: Optional[int]  # Thermal design power in watts
    board_length_mm: Optional[float]  # Length of the GPU board in millimeters
    board_width_mm: Optional[float]  # Width of the GPU board in millimeters
    board_slot_width: Optional[str]  # The number of slots the GPU occupies
    suggested_psu_w: Optional[int]  # Suggested power supply unit in watts
    power_connectors: Optional[str]  # Power connectors required by the GPU, variant
    display_connectors: Optional[str]  # Display connectors available on the GPU, variant
    directx_major_version: Optional[int]  # DirectX major version supported by the GPU
    directx_minor_version: Optional[int]  # DirectX minor version supported by the GPU
    opengl_major_version: Optional[int]  # OpenGL major version supported by the GPU
    opengl_minor_version: Optional[int]  # OpenGL minor version supported by the GPU
    vulkan_major_version: Optional[int]  # Vulkan major version supported by the GPU
    vulkan_minor_version: Optional[int]  # Vulkan minor version supported by the GPU
    opencl_major_version: Optional[int]  # OpenCL major version supported by the GPU
    opencl_minor_version: Optional[int]  # OpenCL minor version supported by the GPU
    cuda_major_version: Optional[int]  # CUDA major version supported by the GPU
    cuda_minor_version: Optional[int]  # CUDA minor version supported by the GPU
    shader_model_major_version: Optional[int]  # Shader model major version supported by the GPU
    shader_model_minor_version: Optional[int]  # Shader model minor version supported by the GPU
    pixel_rate_gpixel_s: Optional[float]  # Pixel fill rate in GPixel/s
    texture_rate_gtexel_s: Optional[float]  # Texture fill rate in GTexel/s
    half_float_performance_gflop_s: Optional[float]  # Half-precision floating point performance in GFLOPS
    single_float_performance_gflop_s: Optional[float]  # Single-precision floating point performance in GFLOPS
    double_float_performance_gflop_s: Optional[float]  # Double-precision floating point performance in GFLOPS
```

To use your own database:

```py
from dbgpu import GPUDatabase
database = GPUDatabase.from_file("path/to/database.json")
```

Supported formats are JSON, CSV and PKL. The PKL format is the fastest to load and is recommended for large databases.

## Command Line

```sh
$ dbgpu lookup "GeForce GTX 1080"
# Using fuzzy search (slower):
# dbgpu lookup GTX1080 --fuzzy

╒═══════════════════════════════════════════════════════════════════════╕
│ GeForce GTX 1080                                                      │
├───────────────────────────────────────────────────────────────────────┤
│                   GPU Name | GP104                                    │
│               Manufacturer | NVIDIA                                   │
│               Architecture | Pascal                                   │
│                    Foundry | TSMC                                     │
│               Process Size | 16 nm                                    │
│           Transistor Count | 7.2 billion                              │
│         Transistor Density | 22.9 million/mm²                         │
│                   Die Size | 314 mm²                                  │
│               Chip Package | BGA-2150                                 │
│               Release Date | 2016-05-27                               │
│                 Generation | GeForce 10                               │
│              Bus Interface | PCIe 3.0 x16                             │
│                 Base Clock | 1,607 MHz                                │
│                Boost Clock | 1,733 MHz                                │
│               Memory Clock | 1,251 MHz                                │
│                Memory Size | 8.0 GB                                   │
│                Memory Type | GDDR5X                                   │
│                 Memory Bus | 256 bit                                  │
│           Memory Bandwidth | 320.3 GB/s                               │
│              Shading Units | 2,560                                    │
│      Texture Mapping Units | 160                                      │
│   Render Output Processors | 64                                       │
│  Streaming Multiprocessors | 20                                       │
│               Tensor Cores | Unknown                                  │
│          Ray Tracing Cores | Unknown                                  │
│                   L1 Cache | 48.0 KB                                  │
│                   L2 Cache | 2.0 MB                                   │
│       Thermal Design Power | 180 W                                    │
│               Board Length | 267 mm                                   │
│                Board Width | 112 mm                                   │
│           Board Slot Width | Dual-slot                                │
│              Suggested PSU | 450 W                                    │
│           Power Connectors | 1x 8-pin                                 │
│         Display Connectors | 1x DVI, 1x HDMI 2.0, 3x DisplayPort 1.4a │
│            DirectX Version | 12.1                                     │
│             OpenGL Version | 4.6                                      │
│             Vulkan Version | 1.3                                      │
│             OpenCL Version | 3.0                                      │
│               CUDA Version | 6.1                                      │
│       Shader Model Version | 6.7                                      │
│                 Pixel Rate | 110.9 GPixel/s                           │
│               Texture Rate | 277.3 GTexel/s                           │
│     Half Float Performance | 138.6 GFLOP/s                            │
│   Single Float Performance | 8.9 TFLOP/s                              │
│   Double Float Performance | 277.3 GFLOP/s                            │
╘═══════════════════════════════════════════════════════════════════════╛
```

This is the output with `tabulate` available; see above for an example without it installed.

Here is a potentially useful bash one-liner to look up the local machine, assuming the availability of the `nvidia-smi` tool:

```bash
dbgpu lookup "$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{$1=""; print $0}' | cut -c2-)"
```

### Building a Database

When installing from PyPI, the latest database is included. If you want to build the database yourself, you can use the `dbgpu` command line tool:

```sh
dbgpu build
```

Note that requests are limited to 4 per minute to be courteous to TechPowerUp's servers. With over 2000 GPUs, **a full build will take over 10 hours,** with most of it spent waiting.

For that reason, if you need to build your own database, it's recommended to limit the build to a specific manufacturer and/or year range, e.g.:

```sh
dbgpu build --manufacturer NVIDIA --start-year 2023
```

Pass `--help` for more options.

```sh
Usage: dbgpu build [OPTIONS]

  Builds a database of GPUs from TechPowerUp.

Options:
  -o, --output PATH           Output file path.  [default: /home/benjamin/mini
                              conda3/envs/enfugue/lib/python3.10/site-
                              packages/dbgpu/data.pkl]
  -m, --manufacturer TEXT     GPU manufacturers to include. Pass multiple
                              times for multiple manufacturers.  [default:
                              NVIDIA, AMD, Intel, ATI, 3dfx, Matrox, XGI,
                              Sony]
  -y, --start-year INTEGER    Start year for GPU database.  [default: 1986]
  -Y, --end-year INTEGER      End year for GPU database.  [default: 2024]
  -d, --courtesy-delay FLOAT  Delay in seconds between requests.  [default:
                              15.0]
  -p, --proxy TEXT            HTTPS proxy URL.
  -t, --timeout FLOAT         Timeout in seconds.
  -r, --retry-max INTEGER     Maximum number of retries.  [default: 5]
  -R, --retry-delay FLOAT     Delay in seconds between retries.  [default:
                              15.0]
  --help                      Show this message and exit.
```

# License

DBGPU is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

# Acknowledgements

This project is not affiliated with [TechPowerUp](https://www.techpowerup.com/gpu-specs/), but could not exist without their website. If you find this project useful, please consider supporting them.
