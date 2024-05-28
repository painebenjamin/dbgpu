import os
import re

from datetime import datetime
from typing import List, Literal

__all__ = [
    "DEFAULT_GPU_DB_PATH",
    "DEFAULT_GPU_DB",
    "DEFAULT_START_YEAR",
    "DEFAULT_END_YEAR",
    "DEFAULT_COURTESY_DELAY",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_RETRY_MAX",
    "MANUFACTURER_LITERAL",
    "ALL_MANUFACTURERS",
    "PARENTHESIZED_VERSION_NUMBER",
    "STANDALONE_VERSION_NUMBER",
    "STANDALONE_NUMBER",
    "NM_SIZE",
    "MM_SIZE",
    "CLOCK_SPEED",
    "BUS_WIDTH",
    "WATTS",
    "TRANSISTOR_COUNT",
    "TRANSISTOR_DENSITY",
    "FLOPS_SIZE",
    "TEXEL_RATE",
    "PIXEL_RATE",
    "BANDWIDTH",
    "BYTES",
    "DATE_YEAR_ONLY",
    "DATE_MONTH_YEAR",
    "DATE_FULL",
    "UNIT_THRESHOLD",
    "NO_DEFAULT",
]

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GPU_DB = None
DEFAULT_GPU_DB_PATH = os.path.join(HERE, "data.pkl")

# Default values
DEFAULT_START_YEAR = 1986
DEFAULT_END_YEAR = datetime.now().year
DEFAULT_COURTESY_DELAY = 15.0
DEFAULT_RETRY_DELAY = 15.0
DEFAULT_RETRY_MAX = 5

# Manufacturers
MANUFACTURER_LITERAL = Literal["NVIDIA", "AMD", "Intel", "ATI", "3dfx", "Matrox", "XGI", "Sony"]
ALL_MANUFACTURERS: List[MANUFACTURER_LITERAL] = ["NVIDIA", "AMD", "Intel", "ATI", "3dfx", "Matrox", "XGI", "Sony"]

# Regular expressions to extract values from strings
PARENTHESIZED_VERSION_NUMBER = re.compile(r"(\((\d+)([._](\d+))?\))")
STANDALONE_VERSION_NUMBER = re.compile(r"((\d+)([._](\d+))?)")
STANDALONE_NUMBER = re.compile(r"((\d+(\.\d+)?))")

# Fixed units
NM_SIZE = re.compile(r"((\d+)\W*nm)")  # always in nm
MM_SIZE = re.compile(r"((\d+)\W*mm(²)?)")  # mm or mm²
CLOCK_SPEED = re.compile(r"((\d+(\.\d+)?)\W*mhz)")  # always in MHz
BUS_WIDTH = re.compile(r"((\d+)\W*bit)")  # always in bits
WATTS = re.compile(r"((\d+)\W*w)")  # always in watts
TRANSISTOR_COUNT = re.compile(r"((\d+)\W*million)")

# Variable units
TRANSISTOR_DENSITY = re.compile(r"((\d+(\.\d+)?)([kmb]\W*\/\W*mm(²)?))")
FLOPS_SIZE = re.compile(r"((\d+(\.\d+)?)\W*([kmgt]flops))")
TEXEL_RATE = re.compile(r"((\d+(\.\d+)?)\W*([kmgt]t(exel)?/s))")
PIXEL_RATE = re.compile(r"((\d+(\.\d+)?)\W*([kmgt]p(ixel)?/s))")
BANDWIDTH = re.compile(r"((\d+(\.\d+)?)\W*([kmgt]b/s))")
BYTES = re.compile(r"((\d+(\.\d+)?)\W*([kmgt]b))")

# Date formats
DATE_YEAR_ONLY = re.compile(r"(\d{4})")
DATE_MONTH_YEAR = re.compile(r"(\w+)\W*(\d{4})")
DATE_FULL = re.compile(r"(\w+)\W*(\d+)[thrnds]*\W*,\W*(\d{4})")

# Threshold for units
# Anything greather than this threshold will be converted to the next unit (e.g. divided by 1000)
# This is purely for aesthetic purposes, GPU manufacturers often choose to display something like
# 1300 GFLOPS instead of 1.3 TFLOPS. We use this threshold to to determine when the unit should be
# converted from the former to the latter.
UNIT_THRESHOLD = 2048

# A class to represent the absence of a default value
class NO_DEFAULT:
    pass
