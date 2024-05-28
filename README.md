# DBGPU

A small, easy-to-use open source database of over 2000 GPUs with architecture, manufacturing, API support and performance details sources from [TechPowerUp](https://www.techpowerup.com/gpu-specs/).

# Installation

DBGPU is available on PyPI and can be installed with pip:

```sh
pip install dbgpu
```

In order to be as minimal as possible, some features are only available as additional dependencies. To install any additional package, use `pip install dbgpu[package]`:
- `dbgpu[tabulate]` will install [tabulate](https://github.com/astanin/python-tabulate/) for pretty-printing tables
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
# spec = database.lookup("GTX 1080")
print(spec)
```

When using your own database:

```py
from dbgpu import GPUDatabase
database = GPUDatabase.from_file("path/to/database.json")
```

Supported formats are JSON, CSV and PKL. The PKL format is the fastest to load and is recommended for large databases.

## Command Line

```sh
dbgpu lookup "GeForce GTX 1080"
# Using fuzzy search (slower):
# dbgpu lookup "GTX 1080" --fuzzy
```

### Building a Database

When installing from PyPI, the latest database is included. If you want to build the database yourself, you can use the `dbgpu` command line tool:

```sh
dbgpu build
```

Note that requests are limited to 4 per minute to be courteous to TechPowerUp's servers. With over 2000 GPUs, **a full build will take over 10 hours,** with most of it spent waiting.

For that reason, if you need to build your own database, it's recommended to limit the build to a specific manufacturer and/or year range, e.g.:

```sh
dbgpu build --manufacturer NVIDIA --year-start 2023
```

Pass `--help` for more options.

# License

DBGPU is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

# Acknowledgements

This project is not affiliated with TechPowerUp. The database is sourced from [TechPowerUp](https://www.techpowerup.com/gpu-specs/), which is a great resource for GPU information. If you find this project useful, please consider supporting them.
