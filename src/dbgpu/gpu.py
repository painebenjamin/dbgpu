from __future__ import annotations

import re

from typing import Tuple, Dict, Any, Optional, List, Union, TYPE_CHECKING

from datetime import datetime, date

from dataclasses import asdict

from dbgpu.constants import *
from dbgpu.util import chunk_iterable, reduce_units, safe_name

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

__all__ = ["GPUSpecification"]

@dataclass
class GPUSpecification:
    """
    A class to represent GPU specifications.
    Parses the details from the TechPowerUp website and stores them in a dictionary.
    """
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

    @classmethod
    def _standardize(cls, value: str) -> str:
        """
        Standardizes a string value.
        """
        return value.replace(",", "").strip().lower()

    @classmethod
    def _is_ignored_value(cls, value: Any) -> bool:
        """
        Checks if a value is one of the ignored values.
        """
        if value is None:
            return True
        if isinstance(value, str):
            return cls._standardize(value) in ["", "none", "n/a", "unknown", "system shared", "system dependent"]
        return False

    @classmethod
    def _get_outputs_from_string(cls, output_string: str) -> Optional[str]:
        """
        Extracts the display outputs from a string.
        """
        if output_string is None or cls._is_ignored_value(output_string):
            return None
        # Split on numbers to get the output types
        # Strings are of the form '1x VGA2x DVI-D1x HDMI1x DisplayPort'
        outputs = re.split(r"(\dx)", output_string)[1:]
        output_labels: List[str] = []
        for amount, output_type in chunk_iterable(outputs, 2):
            output_labels.append(f"{amount.strip()} {output_type.strip()}")
        return ", ".join(output_labels)

    @classmethod
    def _get_version_from_string(cls, version_string: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        """
        Extracts the major and minor version from a string.
        """
        if version_string is None or cls._is_ignored_value(version_string):
            return None, None

        # Look for a parenthesized version number in the version string
        match = PARENTHESIZED_VERSION_NUMBER.search(version_string)
        if match:
            version_major = int(match.group(2))
            version_minor = int(match.group(4)) if match.group(4) else 0
            return version_major, version_minor

        # No parenthesized version number found, look for a standalone version number
        match = STANDALONE_VERSION_NUMBER.search(version_string)
        if not match:
            raise ValueError(
                f"Could not extract version number from '{version_string}'"
            )

        version_major = int(match.group(2))
        version_minor = int(match.group(4)) if match.group(4) else 0
        return version_major, version_minor

    @classmethod
    def _get_mm_from_string(cls, size_string: Optional[str]) -> Optional[int]:
        """
        Extracts the number of millimeters from a size string as found in the TechPowerUp data.
        """
        if size_string is None or cls._is_ignored_value(size_string):
            return None

        match = MM_SIZE.search(cls._standardize(size_string))
        if not match:
            raise ValueError(f"Could not extract millimeters from '{size_string}'")

        return int(match.group(2))

    @classmethod
    def _get_gigaflops_from_string(cls, flops_string: Optional[str]) -> Optional[float]:
        """
        Extracts the number of TFLOPS from a string.
        """
        if flops_string is None or cls._is_ignored_value(flops_string):
            return None

        match = FLOPS_SIZE.search(cls._standardize(flops_string))
        if not match:
            raise ValueError(f"Could not extract FLOPS from '{flops_string}'")
        flops = float(match.group(2))
        unit = match.group(4)[0]

        if unit == "k":
            flops *= 1e3
        if unit == "m":
            flops *= 1e6
        if unit == "g":
            flops *= 1e9
        if unit == "t":
            flops *= 1e12

        # Convert back to gigaflops
        return flops / 1e9

    @classmethod
    def _get_bytes_from_string(cls, size_string: Optional[str], dividend: Union[int,float]=1) -> float:
        """
        Extracts the number of megabytes from a size string.
        """
        if size_string is None or cls._is_ignored_value(size_string):
            return 0.0

        match = BYTES.search(cls._standardize(size_string))
        if not match:
            raise ValueError(f"Could not extract megabytes from '{size_string}'")

        num_bytes = float(match.group(2))
        # Sometimes TPU interchanges base 2 and base 10, we use 10
        if num_bytes % 256 == 0:
            num_bytes /= 256 * 250 # base 2 to base 10

        unit = match.group(4)[0]

        if unit == "k":
            num_bytes *= 1e3
        if unit == "m":
            num_bytes *= 1e6
        if unit == "g":
            num_bytes *= 1e9
        if unit == "t":
            num_bytes *= 1e12

        return num_bytes / dividend

    @classmethod
    def _get_kilobytes_from_string(cls, size_string: Optional[str]) -> float:
        """
        Extracts the number of kilobytes from a size string.
        """
        return cls._get_bytes_from_string(size_string, 1e3)

    @classmethod
    def _get_megabytes_from_string(cls, size_string: Optional[str]) -> float:
        """
        Extracts the number of megabytes from a size string.
        """
        return cls._get_bytes_from_string(size_string, 1e6)

    @classmethod
    def _get_gigabytes_from_string(cls, size_string: Optional[str]) -> float:
        """
        Extracts the number of gigabytes from a size string.
        """
        return cls._get_bytes_from_string(size_string, 1e9)

    @classmethod
    def _get_gigatexels_from_string(cls, texel_string: Optional[str]) -> Optional[float]:
        """
        Extracts the number of GTexels/s from a string.
        """
        if texel_string is None or cls._is_ignored_value(texel_string):
            return None

        match = TEXEL_RATE.search(cls._standardize(texel_string))
        if not match:
            raise ValueError(f"Could not extract GTexels/s from '{texel_string}'")
        texels = float(match.group(2))
        unit = match.group(4)[0]

        if unit == "k":
            texels *= 1e3
        if unit == "m":
            texels *= 1e6
        if unit == "g":
            texels *= 1e9
        if unit == "t":
            texels *= 1e12

        # Convert back to gigatexels
        return texels / 1e9

    @classmethod
    def _get_gigapixels_from_string(cls, pixel_string: Optional[str]) -> Optional[float]:
        """
        Extracts the number of GPixels/s from a string.
        """
        if pixel_string is None or cls._is_ignored_value(pixel_string):
            return None

        match = PIXEL_RATE.search(cls._standardize(pixel_string))
        if not match:
            raise ValueError(f"Could not extract GPixels/s from '{pixel_string}'")
        pixels = float(match.group(2))
        unit = match.group(4)[0]

        if unit == "k":
            pixels *= 1e3
        if unit == "m":
            pixels *= 1e6
        if unit == "g":
            pixels *= 1e9
        if unit == "t":
            pixels *= 1e12

        # Convert back to gigapixels
        return pixels / 1e9

    @classmethod
    def _get_gb_s_from_string(cls, bandwidth_string: Optional[str]) -> Optional[float]:
        """
        Extracts the number of GB/s from a string.
        """
        if bandwidth_string is None or cls._is_ignored_value(bandwidth_string):
            return None

        match = BANDWIDTH.search(cls._standardize(bandwidth_string))
        if not match:
            raise ValueError(f"Could not extract GB/s from '{bandwidth_string}'")
        bandwidth = float(match.group(2))
        unit = match.group(4)[0]

        if unit == "k":
            bandwidth *= 1e3
        if unit == "m":
            bandwidth *= 1e6
        if unit == "g":
            bandwidth *= 1e9
        if unit == "t":
            bandwidth *= 1e12

        # Convert back to gigabytes
        return bandwidth / 1e9

    @classmethod
    def _get_transistor_density_from_string(cls, density_string: Optional[str]) -> Optional[float]:
        """
        Extracts the transistor density from a string (in million transistors per square millimeter).
        """
        if density_string is None or cls._is_ignored_value(density_string):
            return None

        match = TRANSISTOR_DENSITY.search(cls._standardize(density_string))
        if not match:
            raise ValueError(
                f"Could not extract transistor density from '{density_string}'"
            )

        density = float(match.group(2))
        unit = match.group(4)[0]

        if unit == "k":
            density *= 1e3
        if unit == "m":
            density *= 1e6
        if unit == "b":
            density *= 1e9

        # Convert back to thousand transistors per square millimeter
        return density / 1e3

    @classmethod
    def _get_transistor_count_from_string(cls, transistor_string: Optional[str]) -> Optional[int]:
        """
        Extracts the number of transistors from a string (in millions).
        """
        if transistor_string is None or cls._is_ignored_value(transistor_string):
            return None

        match = TRANSISTOR_COUNT.search(cls._standardize(transistor_string))
        if not match:
            raise ValueError(
                f"Could not extract transistor count from '{transistor_string}'"
            )

        return int(match.group(2))

    @classmethod
    def _get_process_size_nm_from_string(cls, size_string: Optional[str]) -> Optional[int]:
        """
        Extracts the process size in nanometers from a string.
        """
        if size_string is None or cls._is_ignored_value(size_string):
            return None

        match = NM_SIZE.search(cls._standardize(size_string))
        if not match:
            raise ValueError(f"Could not extract nanometers from '{size_string}'")

        return int(match.group(2))

    @classmethod
    def _get_clock_speed_from_string(cls, clock_string: Optional[str]) -> Optional[float]:
        """
        Extracts the clock speed in MHz from a string.
        """
        if clock_string is None or cls._is_ignored_value(clock_string):
            return None

        match = CLOCK_SPEED.search(cls._standardize(clock_string))
        if not match:
            raise ValueError(f"Could not extract MHz from '{clock_string}'")

        return float(match.group(2))

    @classmethod
    def _get_bus_width_from_string(cls, bus_string: Optional[str]) -> Optional[int]:
        """
        Extracts the bus width in bits from a string.
        """
        if bus_string is None or cls._is_ignored_value(bus_string):
            return None

        match = BUS_WIDTH.search(cls._standardize(bus_string))
        if not match:
            raise ValueError(f"Could not extract bits from '{bus_string}'")

        return int(match.group(2))

    @classmethod
    def _get_watts_from_string(cls, watts_string: Optional[str]) -> Optional[int]:
        """
        Extracts the number of watts from a string.
        """
        if watts_string is None or cls._is_ignored_value(watts_string):
            return None

        match = WATTS.search(cls._standardize(watts_string))
        if not match:
            raise ValueError(f"Could not extract watts from '{watts_string}'")

        return int(match.group(2))

    @classmethod
    def _get_release_date_from_string(cls, date_string: Optional[str]) -> Optional[date]:
        """
        Extracts the release date from a string.
        """
        if date_string is None or cls._is_ignored_value(date_string) or date_string == "Never Released":
            return None

        # Check for a full date
        match = DATE_FULL.search(date_string)
        if match:
            day = int(match.group(2).strip("thrdns"))
            month = match.group(1)
            year = int(match.group(3))
            return date(year, datetime.strptime(month, "%b").month, day)

        # Check for a month and year date
        match = DATE_MONTH_YEAR.search(date_string)
        if match:
            month = match.group(1)
            year = int(match.group(2))
            return date(year, datetime.strptime(month, "%b").month, 1)

        # Check for a year-only date
        match = DATE_YEAR_ONLY.search(date_string)
        if match:
            return date(int(match.group(1)), 1, 1)

        raise ValueError(f"Could not extract date from '{date_string}'")

    @classmethod
    def _get_number_from_string(cls, number_string: Optional[str]) -> float:
        """
        Extracts a number from a string.
        This is a generic method that can be used to extract any number from a string,
        but it is not recommended to use it for specific types of numbers like clock speeds.
        """
        if number_string is None or cls._is_ignored_value(number_string):
            return 0.0

        match = STANDALONE_NUMBER.search(cls._standardize(number_string))
        if not match:
            raise ValueError(f"Could not extract number from '{number_string}'")

        return float(match.group(1))

    @classmethod
    def _get_integer_from_string(cls, number_string: Optional[str]) -> int:
        """
        Extracts an integer from a string.
        """
        return int(cls._get_number_from_string(number_string))

    @classmethod
    def _get_multikey(cls, dictionary: Dict[str, Any], *keys: str, default: Any = NO_DEFAULT) -> Any:
        """
        Retrieves a value from a dictionary using multiple keys.
        """
        for key in keys:
            if key in dictionary:
                return dictionary[key]
        if default is NO_DEFAULT:
            raise KeyError(f"None of the keys {keys} found in dictionary")
        return default

    @classmethod
    def from_tpu_dict(
        cls,
        name: str,
        manufacturer: MANUFACTURER_LITERAL,
        data: Dict[str, Dict[str, Any]]
    ) -> GPUSpecification:
        """
        Creates a GPUSpecification object from a TechPowerUp dictionary.
        """
        # First grab each section of the data
        try:
            graphics_processor = data["Graphics Processor"]
            graphics_card = cls._get_multikey(data, "Graphics Card", "Mobile Graphics", "Integrated Graphics")
            memory = data["Memory"]
            clock_speeds = data["Clock Speeds"]
            render_config = data["Render Config"]
            performance = data["Theoretical Performance"]
            board = data["Board Design"]
            graphics_features = data["Graphics Features"]
        except KeyError as e:
            raise ValueError(f"Missing key {e} in data dictionary")

        api_support: Dict[str, Dict[str, Optional[int]]] = {}

        # Split major and minor versions for API support
        for api in ["DirectX", "OpenGL", "Vulkan", "OpenCL", "CUDA", "Shader Model"]:
            support_str = graphics_features.get(api, "").strip()
            if not support_str:
                api_support[api] = {"Major": None, "Minor": None}
                continue
            major, minor = cls._get_version_from_string(cls._standardize(support_str))
            api_support[api] = {"Major": major, "Minor": minor}

        # Now go through the sections and split them into the individual fields
        return cls(
            manufacturer=manufacturer,
            name=name,
            gpu_name=graphics_processor["GPU Name"],
            architecture=graphics_processor.get("Architecture", None),
            foundry=graphics_processor.get("Foundry", None),
            process_size_nm=cls._get_process_size_nm_from_string(
                graphics_processor["Process Size"]
            ),
            transistor_count_m=cls._get_transistor_count_from_string(
                graphics_processor["Transistors"]
            ),
            transistor_density_k_mm2=cls._get_transistor_density_from_string(
                graphics_processor.get("Density", None)
            ),
            die_size_mm2=cls._get_mm_from_string(
                graphics_processor["Die Size"]
            ),
            chip_package=graphics_processor.get("Chip Package", None),
            release_date=cls._get_release_date_from_string(
                graphics_card["Release Date"]
            ),
            generation=graphics_card["Generation"],
            bus_interface=graphics_card.get("Bus Interface", None),
            base_clock_mhz=cls._get_clock_speed_from_string(
                cls._get_multikey(clock_speeds, "Base Clock", "GPU Clock")
            ),
            boost_clock_mhz=cls._get_clock_speed_from_string(
                cls._get_multikey(clock_speeds, "Boost Clock", "GPU Clock")
            ),
            memory_clock_mhz=cls._get_clock_speed_from_string(
                clock_speeds["Memory Clock"]
            ),
            memory_size_gb=cls._get_gigabytes_from_string(
                memory["Memory Size"]
            ),
            memory_type=memory["Memory Type"],
            memory_bus_bits=cls._get_bus_width_from_string(
                memory["Memory Bus"]
            ),
            memory_bandwidth_gb_s=cls._get_gb_s_from_string(
                memory["Bandwidth"]
            ),
            shading_units=cls._get_integer_from_string(
                render_config.get("Shading Units", None)
            ),
            texture_mapping_units=cls._get_integer_from_string(
                render_config["TMUs"]
            ),
            render_output_processors=cls._get_integer_from_string(
                render_config["ROPs"]
            ),
            streaming_multiprocessors=cls._get_integer_from_string(
                cls._get_multikey(render_config, "SMM Count", "SM Count", default=None)
            ),
            tensor_cores=cls._get_integer_from_string(
                render_config.get("Tensor Cores", None)
            ),
            ray_tracing_cores=cls._get_integer_from_string(
                render_config.get("RT Cores", None)
            ),
            l1_cache_kb=cls._get_kilobytes_from_string(
                render_config.get("L1 Cache", None)
            ),
            l2_cache_mb=cls._get_megabytes_from_string(
                render_config.get("L2 Cache", None)
            ),
            thermal_design_power_w=cls._get_watts_from_string(
                board["TDP"]
            ),
            board_length_mm=cls._get_mm_from_string(
                board.get("Length", None)
            ),
            board_width_mm=cls._get_mm_from_string(
                board.get("Width", None)
            ),
            board_slot_width=board.get("Slot Width", None),
            suggested_psu_w=cls._get_watts_from_string(
                board.get("Suggested PSU", None)
            ),
            power_connectors=board.get("Power Connectors", None),
            display_connectors=cls._get_outputs_from_string(board["Outputs"]),
            directx_major_version=api_support["DirectX"]["Major"],
            directx_minor_version=api_support["DirectX"]["Minor"],
            opengl_major_version=api_support["OpenGL"]["Major"],
            opengl_minor_version=api_support["OpenGL"]["Minor"],
            vulkan_major_version=api_support["Vulkan"]["Major"],
            vulkan_minor_version=api_support["Vulkan"]["Minor"],
            opencl_major_version=api_support["OpenCL"]["Major"],
            opencl_minor_version=api_support["OpenCL"]["Minor"],
            cuda_major_version=api_support["CUDA"]["Major"],
            cuda_minor_version=api_support["CUDA"]["Minor"],
            shader_model_major_version=api_support["Shader Model"]["Major"],
            shader_model_minor_version=api_support["Shader Model"]["Minor"],
            pixel_rate_gpixel_s=cls._get_gigapixels_from_string(
                performance["Pixel Rate"]
            ),
            texture_rate_gtexel_s=cls._get_gigatexels_from_string(
                performance["Texture Rate"]
            ),
            half_float_performance_gflop_s=cls._get_gigaflops_from_string(
                performance.get("FP16 (half)", None)
            ),
            single_float_performance_gflop_s=cls._get_gigaflops_from_string(
                performance.get("FP32 (float)", None)
            ),
            double_float_performance_gflop_s=cls._get_gigaflops_from_string(
                performance.get("FP64 (double)", None)
            ),
        )
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the GPU specification as a dictionary.
        """
        return asdict(self)

    @property
    def process_size_str(self) -> str:
        """
        Returns the process size as a formatted string.
        """
        return "Unknown" if not self.process_size_nm else f"{self.process_size_nm} nm"

    @property
    def die_size_str(self) -> str:
        """
        Returns the die size as a formatted string.
        """
        return "Unknown" if not self.die_size_mm2 else f"{self.die_size_mm2:.0f} mm²"

    @property
    def transistor_count_str(self) -> str:
        """
        Returns the transistor count as a formatted string.
        """
        count = self.transistor_count_m
        if not count:
            return "Unknown"
        value, unit = reduce_units(
            count,
            ["million", "billion", "trillion"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}"

    @property
    def transistor_density_str(self) -> str:
        """
        Returns the transistor density as a formatted string.
        """
        density = self.transistor_density_k_mm2
        if not density:
            return "Unknown"
        value, unit = reduce_units(
            density,
            ["thousand", "million", "billion"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}/mm²"

    @property
    def base_clock_str(self) -> str:
        """
        Returns the base clock speed as a formatted string.
        """
        return f"{self.base_clock_mhz:,.0f} MHz"

    @property
    def boost_clock_str(self) -> str:
        """
        Returns the boost clock speed as a formatted string.
        """
        return f"{self.boost_clock_mhz:,.0f} MHz"

    @property
    def memory_clock_str(self) -> str:
        """
        Returns the memory clock speed as a formatted string.
        """
        return "Unknown" if not self.memory_clock_mhz else f"{self.memory_clock_mhz:,.0f} MHz"

    @property
    def memory_size_str(self) -> str:
        """
        Returns the memory size as a formatted string.
        """
        return "Unknown" if not self.memory_size_gb else f"{self.memory_size_gb:0,.1f} GB"

    @property
    def memory_bus_str(self) -> str:
        """
        Returns the memory bus width as a formatted string.
        """
        return "Unknown" if not self.memory_bus_bits else f"{self.memory_bus_bits} bit"

    @property
    def memory_bandwidth_str(self) -> str:
        """
        Returns the memory bandwidth as a formatted string.
        """
        return "Unknown" if not self.memory_bandwidth_gb_s else f"{self.memory_bandwidth_gb_s:0,.1f} GB/s"

    @property
    def l1_cache_str(self) -> str:
        """
        Returns the L1 cache size as a formatted string.
        """
        return f"{self.l1_cache_kb:,.1f} KB"

    @property
    def l2_cache_str(self) -> str:
        """
        Returns the L2 cache size as a formatted string.
        """
        return f"{self.l2_cache_mb:,.1f} MB"

    @property
    def thermal_design_power_str(self) -> str:
        """
        Returns the thermal design power as a formatted string.
        """
        return "Unknown" if not self.thermal_design_power_w else f"{self.thermal_design_power_w} W"

    @property
    def board_length_str(self) -> str:
        """
        Returns the board length as a formatted string.
        """
        return "Unknown" if not self.board_length_mm else f"{self.board_length_mm:.0f} mm"

    @property
    def board_width_str(self) -> str:
        """
        Returns the board width as a formatted string.
        """
        return "Unknown" if not self.board_width_mm else f"{self.board_width_mm:.0f} mm"

    @property
    def suggested_psu_str(self) -> str:
        """
        Returns the suggested power supply unit as a formatted string.
        """
        return "Unknown" if not self.suggested_psu_w else f"{self.suggested_psu_w} W"

    @property
    def pixel_rate_str(self) -> str:
        """
        Returns the pixel fill rate as a formatted string.
        """
        if not self.pixel_rate_gpixel_s:
            return "Unknown"
        value, unit = reduce_units(
            self.pixel_rate_gpixel_s,
            ["GPixel", "TPixel", "PPixel", "EPixel", "ZPixel", "YPixel"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}/s"

    @property
    def texture_rate_str(self) -> str:
        """
        Returns the texture fill rate as a formatted string.
        """
        if not self.texture_rate_gtexel_s:
            return "Unknown"
        value, unit = reduce_units(
            self.texture_rate_gtexel_s,
            ["GTexel", "TTexel", "PTexel", "ETexel", "ZTexel", "YTexel"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}/s"

    @property
    def half_float_performance_str(self) -> str:
        """
        Returns the half-precision floating point performance as a formatted string.
        """
        if not self.half_float_performance_gflop_s:
            return "Unknown"
        value, unit = reduce_units(
            self.half_float_performance_gflop_s,
            ["GFLOP", "TFLOP", "PFLOP", "EFLOP", "ZFLOP", "YFLOP"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}/s"

    @property
    def single_float_performance_str(self) -> str:
        """
        Returns the single-precision floating point performance as a formatted string.
        """
        if not self.single_float_performance_gflop_s:
            return "Unknown"
        value, unit = reduce_units(
            self.single_float_performance_gflop_s,
            ["GFLOP", "TFLOP", "PFLOP", "EFLOP", "ZFLOP", "YFLOP"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}/s"

    @property
    def double_float_performance_str(self) -> str:
        """
        Returns the double-precision floating point performance as a formatted string.
        """
        if not self.double_float_performance_gflop_s:
            return "Unknown"
        value, unit = reduce_units(
            self.double_float_performance_gflop_s,
            ["GFLOP", "TFLOP", "PFLOP", "EFLOP", "ZFLOP", "YFLOP"],
            threshold=UNIT_THRESHOLD,
        )
        return f"{value:,.1f} {unit}/s"

    @property
    def directx_version_str(self) -> str:
        """
        Returns the DirectX version as a formatted string.
        """
        return "None" if not self.directx_major_version else f"{self.directx_major_version}.{self.directx_minor_version}"

    @property
    def opengl_version_str(self) -> str:
        """
        Returns the OpenGL version as a formatted string.
        """
        return "None" if not self.opengl_major_version else f"{self.opengl_major_version}.{self.opengl_minor_version}"

    @property
    def vulkan_version_str(self) -> str:
        """
        Returns the Vulkan version as a formatted string.
        """
        return "None" if not self.vulkan_major_version else f"{self.vulkan_major_version}.{self.vulkan_minor_version}"

    @property
    def opencl_version_str(self) -> str:
        """
        Returns the OpenCL version as a formatted string.
        """
        return "None" if not self.opencl_major_version else f"{self.opencl_major_version}.{self.opencl_minor_version}"

    @property
    def cuda_version_str(self) -> str:
        """
        Returns the CUDA version as a formatted string.
        """
        return "None" if not self.cuda_major_version else f"{self.cuda_major_version}.{self.cuda_minor_version}"

    @property
    def shader_model_version_str(self) -> str:
        """
        Returns the shader model version as a formatted string.
        """
        return "None" if not self.shader_model_major_version else f"{self.shader_model_major_version}.{self.shader_model_minor_version}"

    @property
    def shading_units_str(self) -> str:
        """
        Returns the number of shading units as a formatted string.
        """
        return "Unknown" if not self.shading_units else f"{self.shading_units:,}"

    @property
    def texture_mapping_units_str(self) -> str:
        """
        Returns the number of texture mapping units as a formatted string.
        """
        return "Unknown" if not self.texture_mapping_units else f"{self.texture_mapping_units:,}"

    @property
    def render_output_processors_str(self) -> str:
        """
        Returns the number of render output processors as a formatted string.
        """
        return "Unknown" if not self.render_output_processors else f"{self.render_output_processors:,}"

    @property
    def streaming_multiprocessors_str(self) -> str:
        """
        Returns the number of streaming multiprocessors as a formatted string.
        """
        return "Unknown" if not self.streaming_multiprocessors else f"{self.streaming_multiprocessors:,}"

    @property
    def tensor_cores_str(self) -> str:
        """
        Returns the number of tensor cores as a formatted string.
        """
        return "Unknown" if not self.tensor_cores else f"{self.tensor_cores:,}"

    @property
    def ray_tracing_cores_str(self) -> str:
        """
        Returns the number of ray tracing cores as a formatted string.
        """
        return "Unknown" if not self.ray_tracing_cores else f"{self.ray_tracing_cores:,}"

    @property
    def name_key(self) -> str:
        """
        Returns the name of the GPU, suitable for use as a key.
        """
        return safe_name(self.name)

    @property
    def manufacturer_prefixed_name_key(self) -> str:
        """
        Returns the GPU name prefixed with the manufacturer.
        """
        return f"{self.manufacturer.lower()}-{self.name_key}"

    def labeled_fields(self) -> List[Tuple[str, Optional[str]]]:
        """
        Returns a list of tuples with the field names and their formatted values.
        """
        return [
            ("GPU Name", self.gpu_name),
            ("Manufacturer", self.manufacturer),
            ("Architecture", self.architecture),
            ("Foundry", self.foundry),
            ("Process Size", self.process_size_str),
            ("Transistor Count", self.transistor_count_str),
            ("Transistor Density", self.transistor_density_str),
            ("Die Size", self.die_size_str),
            ("Chip Package", self.chip_package),
            ("Release Date", str(self.release_date)),
            ("Generation", self.generation),
            ("Bus Interface", self.bus_interface),
            ("Base Clock", self.base_clock_str),
            ("Boost Clock", self.boost_clock_str),
            ("Memory Clock", self.memory_clock_str),
            ("Memory Size", self.memory_size_str),
            ("Memory Type", self.memory_type),
            ("Memory Bus", self.memory_bus_str),
            ("Memory Bandwidth", self.memory_bandwidth_str),
            ("Shading Units", self.shading_units_str),
            ("Texture Mapping Units", self.texture_mapping_units_str),
            ("Render Output Processors", self.render_output_processors_str),
            ("Streaming Multiprocessors", self.streaming_multiprocessors_str),
            ("Tensor Cores", self.tensor_cores_str),
            ("Ray Tracing Cores", self.ray_tracing_cores_str),
            ("L1 Cache", self.l1_cache_str),
            ("L2 Cache", self.l2_cache_str),
            ("Thermal Design Power", self.thermal_design_power_str),
            ("Board Length", self.board_length_str),
            ("Board Width", self.board_width_str),
            ("Board Slot Width", self.board_slot_width),
            ("Suggested PSU", self.suggested_psu_str),
            ("Power Connectors", self.power_connectors),
            ("Display Connectors", self.display_connectors),
            ("DirectX Version", self.directx_version_str),
            ("OpenGL Version", self.opengl_version_str),
            ("Vulkan Version", self.vulkan_version_str),
            ("OpenCL Version", self.opencl_version_str),
            ("CUDA Version", self.cuda_version_str),
            ("Shader Model Version", self.shader_model_version_str),
            ("Pixel Rate", self.pixel_rate_str),
            ("Texture Rate", self.texture_rate_str),
            ("Half Float Performance", self.half_float_performance_str),
            ("Single Float Performance", self.single_float_performance_str),
            ("Double Float Performance", self.double_float_performance_str),
        ]

    def labeled_comparison_fields(self, other: GPUSpecification) -> List[Tuple[str, str]]:
        """
        Returns a list of tuples with the field names and their formatted values for comparison with another GPU.
        """
        name_difference_str = "!=" if self.gpu_name != other.gpu_name else ""
        manufacturer_difference_str = "!=" if self.manufacturer != other.manufacturer else ""
        architecture_difference_str = "!=" if self.architecture != other.architecture else ""
        foundry_difference_str = "!=" if self.foundry != other.foundry else ""
        chip_package_difference_str = "!=" if self.chip_package != other.chip_package else ""
        release_date_difference_str = "!=" if self.release_date != other.release_date else ""
        generation_difference_str = "!=" if self.generation != other.generation else ""
        bus_interface_difference_str = "!=" if self.bus_interface != other.bus_interface else ""
        directx_version_difference_str = "!=" if self.directx_version_str != other.directx_version_str else ""
        opengl_version_difference_str = "!=" if self.opengl_version_str != other.opengl_version_str else ""
        vulkan_version_difference_str = "!=" if self.vulkan_version_str != other.vulkan_version_str else ""
        opencl_version_difference_str = "!=" if self.opencl_version_str != other.opencl_version_str else ""
        cuda_version_difference_str = "!=" if self.cuda_version_str != other.cuda_version_str else ""
        shader_model_version_difference_str = "!=" if self.shader_model_version_str != other.shader_model_version_str else ""
        power_connectors_difference_str = "!=" if self.power_connectors != other.power_connectors else ""
        display_connectors_difference_str = "!=" if self.display_connectors != other.display_connectors else ""
        memory_type_difference_str = "!=" if self.memory_type != other.memory_type else ""

        if self.process_size_nm and other.process_size_nm:
            process_size_difference = other.process_size_nm - self.process_size_nm
            process_size_difference_str = "{:+d} nm ({:+.1%})".format(
                process_size_difference, process_size_difference / self.process_size_nm
            ) if process_size_difference else ""
        else:
            process_size_difference_str = ""

        if self.transistor_count_m and other.transistor_count_m:
            transistor_count_difference = other.transistor_count_m - self.transistor_count_m
            value, unit = reduce_units(
                transistor_count_difference,
                ["million", "billion", "trillion"],
                threshold=UNIT_THRESHOLD,
            )
            transistor_count_difference_str = "{:+.1f} {:s} ({:+.1%})".format(
                value, unit, transistor_count_difference / self.transistor_count_m
            ) if transistor_count_difference else ""
        else:
            transistor_count_difference_str = ""

        if self.transistor_density_k_mm2 and other.transistor_density_k_mm2:
            transistor_density_difference = other.transistor_density_k_mm2 - self.transistor_density_k_mm2
            value, unit = reduce_units(
                transistor_density_difference,
                ["thousand", "million", "billion"],
                threshold=UNIT_THRESHOLD,
            )
            transistor_density_difference_str = "{:+.1f} {:s}/mm² ({:+.1%})".format(
                value, unit, transistor_density_difference / self.transistor_density_k_mm2
            ) if transistor_density_difference else ""
        else:
            transistor_density_difference_str = ""

        if self.die_size_mm2 and other.die_size_mm2:
            die_size_difference = other.die_size_mm2 - self.die_size_mm2
            die_size_difference_str = "{:+.1f} mm² ({:+.1%})".format(
                die_size_difference, die_size_difference / self.die_size_mm2
            ) if die_size_difference else ""
        else:
            die_size_difference_str = ""

        if self.base_clock_mhz and other.base_clock_mhz:
            base_clock_difference = other.base_clock_mhz - self.base_clock_mhz
            base_clock_difference_str = "{:+.1f} MHz ({:+.1%})".format(
                base_clock_difference, base_clock_difference / self.base_clock_mhz
            ) if base_clock_difference else ""
        else:
            base_clock_difference_str = ""

        if self.boost_clock_mhz and other.boost_clock_mhz:
            boost_clock_difference = other.boost_clock_mhz - self.boost_clock_mhz
            boost_clock_difference_str = "{:+.1f} MHz ({:+.1%})".format(
                boost_clock_difference, boost_clock_difference / self.boost_clock_mhz
            ) if boost_clock_difference else ""
        else:
            boost_clock_difference_str = ""

        if self.memory_clock_mhz and other.memory_clock_mhz:
            memory_clock_difference = other.memory_clock_mhz - self.memory_clock_mhz
            memory_clock_difference_str = "{:+.1f} MHz ({:+.1%})".format(
                memory_clock_difference, memory_clock_difference / self.memory_clock_mhz
            ) if memory_clock_difference else ""
        else:
            memory_clock_difference_str = ""

        if self.memory_size_gb and other.memory_size_gb:
            memory_size_difference = other.memory_size_gb - self.memory_size_gb
            memory_size_difference_str = "{:+.1f} GB ({:+.1%})".format(
                memory_size_difference, memory_size_difference / self.memory_size_gb
            ) if memory_size_difference else ""
        else:
            memory_size_difference_str = ""

        if self.memory_bus_bits and other.memory_bus_bits:
            memory_bus_difference = other.memory_bus_bits - self.memory_bus_bits
            memory_bus_difference_str = "{:+d} bit ({:+.1%})".format(
                memory_bus_difference, memory_bus_difference / self.memory_bus_bits
            ) if memory_bus_difference else ""
        else:
            memory_bus_difference_str = ""

        if self.memory_bandwidth_gb_s and other.memory_bandwidth_gb_s:
            memory_bandwidth_difference = other.memory_bandwidth_gb_s - self.memory_bandwidth_gb_s
            memory_bandwidth_difference_str = "{:+.1f} GB/s ({:+.1%})".format(
                memory_bandwidth_difference, memory_bandwidth_difference / self.memory_bandwidth_gb_s
            ) if memory_bandwidth_difference else ""
        else:
            memory_bandwidth_difference_str = ""

        if self.shading_units and other.shading_units:
            shading_units_difference = other.shading_units - self.shading_units
            shading_units_difference_str = "{:+d} ({:+.1%})".format(
                shading_units_difference, shading_units_difference / self.shading_units
            ) if shading_units_difference else ""
        else:
            shading_units_difference_str = ""

        if self.texture_mapping_units and other.texture_mapping_units:
            texture_mapping_units_difference = other.texture_mapping_units - self.texture_mapping_units
            texture_mapping_units_difference_str = "{:+d} ({:+.1%})".format(
                texture_mapping_units_difference, texture_mapping_units_difference / self.texture_mapping_units
            ) if texture_mapping_units_difference else ""
        else:
            texture_mapping_units_difference_str = ""

        if self.render_output_processors and other.render_output_processors:
            render_output_processors_difference = other.render_output_processors - self.render_output_processors
            render_output_processors_difference_str = "{:+d} ({:+.1%})".format(
                render_output_processors_difference, render_output_processors_difference / self.render_output_processors
            ) if render_output_processors_difference else ""
        else:
            render_output_processors_difference_str = ""

        if self.streaming_multiprocessors and other.streaming_multiprocessors:
            streaming_multiprocessors_difference = other.streaming_multiprocessors - self.streaming_multiprocessors
            streaming_multiprocessors_difference_str = "{:+d} ({:+.1%})".format(
                streaming_multiprocessors_difference, streaming_multiprocessors_difference / self.streaming_multiprocessors
            ) if streaming_multiprocessors_difference else ""
        else:
            streaming_multiprocessors_difference_str = ""

        if self.tensor_cores and other.tensor_cores:
            tensor_cores_difference = other.tensor_cores - self.tensor_cores
            if tensor_cores_difference and self.tensor_cores:
                tensor_cores_difference_str = "{:+d} ({:+.1%})".format(
                    tensor_cores_difference, tensor_cores_difference / self.tensor_cores
                ) if tensor_cores_difference else ""
            else:
                tensor_cores_difference_str = ""
        else:
            tensor_cores_difference_str = ""

        if self.ray_tracing_cores and other.ray_tracing_cores:
            ray_tracing_cores_difference = other.ray_tracing_cores - self.ray_tracing_cores
            ray_tracing_cores_difference_str = "{:+d} ({:+.1%})".format(
                ray_tracing_cores_difference, ray_tracing_cores_difference / self.ray_tracing_cores
            ) if ray_tracing_cores_difference else ""
        else:
            ray_tracing_cores_difference_str = ""

        if self.l1_cache_kb and other.l1_cache_kb:
            l1_cache_difference = other.l1_cache_kb - self.l1_cache_kb
            l1_cache_difference_str = "{:+.1f} KB ({:+.1%})".format(
                l1_cache_difference, l1_cache_difference / self.l1_cache_kb
            ) if l1_cache_difference else ""
        else:
            l1_cache_difference_str = ""

        if self.l2_cache_mb and other.l2_cache_mb:
            l2_cache_difference = other.l2_cache_mb - self.l2_cache_mb
            l2_cache_difference_str = "{:+.1f} MB ({:+.1%})".format(
                l2_cache_difference, l2_cache_difference / self.l2_cache_mb
            ) if l2_cache_difference else ""
        else:
            l2_cache_difference_str = ""

        if self.thermal_design_power_w and other.thermal_design_power_w:
            thermal_design_power_difference = other.thermal_design_power_w - self.thermal_design_power_w
            thermal_design_power_difference_str = "{:+d} W ({:+.1%})".format(
                thermal_design_power_difference, thermal_design_power_difference / self.thermal_design_power_w
            ) if thermal_design_power_difference else ""
        else:
            thermal_design_power_difference_str = ""

        if other.board_length_mm and self.board_length_mm:
            board_length_difference = other.board_length_mm - self.board_length_mm
            board_length_difference_str = "{:+.1f} mm ({:+.1%})".format(
                board_length_difference, board_length_difference / self.board_length_mm
            ) if board_length_difference else ""
        else:
            board_length_difference_str = ""

        if other.board_width_mm and self.board_width_mm:
            board_width_difference = other.board_width_mm - self.board_width_mm
            board_width_difference_str = "{:+.1f} mm ({:+.1%})".format(
                board_width_difference, board_width_difference / self.board_width_mm
            ) if board_width_difference else ""
        else:
            board_width_difference_str = ""

        if other.suggested_psu_w and self.suggested_psu_w:
            suggested_psu_difference = other.suggested_psu_w - self.suggested_psu_w
            suggested_psu_difference_str = "{:+d} W ({:+.1%})".format(
                suggested_psu_difference, suggested_psu_difference / self.suggested_psu_w
            ) if suggested_psu_difference else ""
        else:
            suggested_psu_difference_str = ""

        if other.pixel_rate_gpixel_s and self.pixel_rate_gpixel_s:
            pixel_rate_difference = other.pixel_rate_gpixel_s - self.pixel_rate_gpixel_s
            pixel_rate_difference_str = "{:+.1f} GPixel/s ({:+.1%})".format(
                pixel_rate_difference, pixel_rate_difference / self.pixel_rate_gpixel_s
            ) if pixel_rate_difference else ""
        else:
            pixel_rate_difference_str = ""

        if other.texture_rate_gtexel_s and self.texture_rate_gtexel_s:
            texture_rate_difference = other.texture_rate_gtexel_s - self.texture_rate_gtexel_s
            texture_rate_difference_str = "{:+.1f} GTexel/s ({:+.1%})".format(
                texture_rate_difference, texture_rate_difference / self.texture_rate_gtexel_s
            ) if texture_rate_difference else ""
        else:
            texture_rate_difference_str = ""

        if other.half_float_performance_gflop_s and self.half_float_performance_gflop_s:
            half_float_performance_difference = other.half_float_performance_gflop_s - self.half_float_performance_gflop_s
            value, unit = reduce_units(
                half_float_performance_difference,
                ["GFLOP", "TFLOP", "PFLOP", "EFLOP", "ZFLOP", "YFLOP"],
                threshold=UNIT_THRESHOLD,
            )
            half_float_performance_difference_str = "{:+.1f} {:s}/s ({:+.1%})".format(
                value, unit, half_float_performance_difference / self.half_float_performance_gflop_s
            ) if half_float_performance_difference else ""
        else:
            half_float_performance_difference_str = ""

        if other.single_float_performance_gflop_s and self.single_float_performance_gflop_s:
            single_float_performance_difference = other.single_float_performance_gflop_s - self.single_float_performance_gflop_s
            value, unit = reduce_units(
                single_float_performance_difference,
                ["GFLOP", "TFLOP", "PFLOP", "EFLOP", "ZFLOP", "YFLOP"],
                threshold=UNIT_THRESHOLD,
            )
            single_float_performance_difference_str = "{:+.1f} {:s}/s ({:+.1%})".format(
                value, unit, single_float_performance_difference / self.single_float_performance_gflop_s
            ) if single_float_performance_difference else ""
        else:
            single_float_performance_difference_str = ""

        if other.double_float_performance_gflop_s and self.double_float_performance_gflop_s:
            double_float_performance_difference = other.double_float_performance_gflop_s - self.double_float_performance_gflop_s
            value, unit = reduce_units(
                double_float_performance_difference,
                ["GFLOP", "TFLOP", "PFLOP", "EFLOP", "ZFLOP", "YFLOP"],
                threshold=UNIT_THRESHOLD,
            )
            double_float_performance_difference_str = "{:+.1f} {:s}/s ({:+.1%})".format(
                value, unit, double_float_performance_difference / self.double_float_performance_gflop_s
            ) if double_float_performance_difference else ""
        else:
            double_float_performance_difference_str = ""

        return [
            ("GPU Name", name_difference_str),
            ("Manufacturer", manufacturer_difference_str),
            ("Architecture", architecture_difference_str),
            ("Foundry", foundry_difference_str),
            ("Process Size", process_size_difference_str),
            ("Transistor Count", transistor_count_difference_str),
            ("Transistor Density", transistor_density_difference_str),
            ("Die Size", die_size_difference_str),
            ("Chip Package", chip_package_difference_str),
            ("Release Date", release_date_difference_str),
            ("Generation", generation_difference_str),
            ("Bus Interface", bus_interface_difference_str),
            ("Base Clock", base_clock_difference_str),
            ("Boost Clock", boost_clock_difference_str),
            ("Memory Clock", memory_clock_difference_str),
            ("Memory Size", memory_size_difference_str),
            ("Memory Type", memory_type_difference_str),
            ("Memory Bus", memory_bus_difference_str),
            ("Memory Bandwidth", memory_bandwidth_difference_str),
            ("Shading Units", shading_units_difference_str),
            ("Texture Mapping Units", texture_mapping_units_difference_str),
            ("Render Output Processors", render_output_processors_difference_str),
            ("Streaming Multiprocessors", streaming_multiprocessors_difference_str),
            ("Tensor Cores", tensor_cores_difference_str),
            ("Ray Tracing Cores", ray_tracing_cores_difference_str),
            ("L1 Cache", l1_cache_difference_str),
            ("L2 Cache", l2_cache_difference_str),
            ("Thermal Design Power", thermal_design_power_difference_str),
            ("Board Length", board_length_difference_str),
            ("Board Width", board_width_difference_str),
            ("Suggested PSU", suggested_psu_difference_str),
            ("Power Connectors", power_connectors_difference_str),
            ("Display Connectors", display_connectors_difference_str),
            ("DirectX Version", directx_version_difference_str),
            ("OpenGL Version", opengl_version_difference_str),
            ("Vulkan Version", vulkan_version_difference_str),
            ("OpenCL Version", opencl_version_difference_str),
            ("CUDA Version", cuda_version_difference_str),
            ("Shader Model Version", shader_model_version_difference_str),
            ("Pixel Rate", pixel_rate_difference_str),
            ("Texture Rate", texture_rate_difference_str),
            ("Half Float Performance", half_float_performance_difference_str),
            ("Single Float Performance", single_float_performance_difference_str),
            ("Double Float Performance", double_float_performance_difference_str),
        ]

    def tabulate(self) -> str:
        """
        Returns a tabulated string representation of the GPU specification.
        """
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        return tabulate.tabulate(
            [
                [self.name],
                [
                    tabulate.tabulate(
                        self.labeled_fields(),
                        tablefmt="presto",
                        colalign=("right", "left"),
                    )
                ]
            ],
            tablefmt="fancy_grid",
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the GPU specification.
        """
        try:
            import tabulate
            return self.tabulate()
        except ImportError:
            return "\n".join([
                "-"*len(self.name),
                self.name,
                "-"*len(self.name),
            ] + [
                f"{label}: {value}"
                for label, value in self.labeled_fields()
            ])
