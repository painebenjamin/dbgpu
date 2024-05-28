from __future__ import annotations

import os
import time
import json
import csv
import pickle
import warnings

from typing import List, Dict, Any, Iterable, Tuple, Optional, TYPE_CHECKING

from dataclasses import dataclass, field

from dbgpu.constants import *
from dbgpu.util import json_serialize

if TYPE_CHECKING:
    from dbgpu.gpu import GPUSpecification
    from requests import Session

@dataclass
class TechPowerUp:
    """
    A class to fetch GPU specs from TechPowerUp.
    """
    base_url: str = "https://www.techpowerup.com"
    list_url: str = "/gpu-specs/"
    manufacturers: List[MANUFACTURER_LITERAL] = field(default_factory=lambda: ALL_MANUFACTURERS)
    timeout: Optional[float] = None
    proxies: Dict[str, str] = field(default_factory=dict)
    retry_max: int = DEFAULT_RETRY_MAX
    retry_delay: float = DEFAULT_RETRY_DELAY

    @property
    def session(self) -> Session:
        """
        A requests session with the specified proxy settings.
        """
        import requests
        if not hasattr(self, "_session"):
            self._session = requests.Session()
            self._session.proxies.update(self.proxies)
        return self._session

    @session.deleter
    def session(self) -> None:
        """
        Deletes the requests session.
        """
        if hasattr(self, "_session"):
            del self._session

    def fetch_gpu_list(
        self,
        manufacturer: MANUFACTURER_LITERAL,
        year: int,
        num_retries: int = 0
    ) -> Iterable[Tuple[str, str]]:
        """
        Fetches a list of GPUs released in a given year and yields a tuple
        of GPU name and link to the details page.

        >>> tpu = TechPowerUp()
        >>> iterator = tpu.fetch_gpu_list("NVIDIA", 2020)
        >>> next(iterator)
        ('A100 PCIe 40 GB', '/gpu-specs/a100-pcie-40-gb.c3623')
        """
        from bs4 import BeautifulSoup
        url = f"{self.base_url}{self.list_url}"
        params = {"mfgr": manufacturer, "released": year}
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except:
            if num_retries < self.retry_max:
                del self.session
                time.sleep(self.retry_delay)
                for gpu in self.fetch_gpu_list(
                    manufacturer=manufacturer,
                    year=year,
                    num_retries=num_retries + 1
                ):
                    yield gpu
                return
            raise

        soup = BeautifulSoup(response.text, "html.parser")
        gpu_table = soup.find("table", {"class": "processors"})

        if not gpu_table:
            warnings.warn(f"No data available for year {year}")
            return

        for row in gpu_table.find_all("tr")[2:]:  # type: ignore[union-attr]
            cells = row.find_all("td")
            name_cell = cells[0]
            link = name_cell.find("a")["href"]
            gpu_name = name_cell.get_text(strip=True)
            yield gpu_name, link

    def fetch_gpu_details(
        self,
        url: str,
        num_retries: int = 0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetches the detailed specs for a given GPU.

        >>> tpu = TechPowerUp()
        >>> specs = tpu.fetch_gpu_details("/gpu-specs/a100-pcie-40-gb.c3623")
        >>> specs["Graphics Processor"]["Architecture"]
        'Ampere'
        """
        from bs4 import BeautifulSoup
        if not url.startswith("http"):
            url = f"{self.base_url}{url}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except:
            if num_retries < self.retry_max:
                del self.session
                time.sleep(self.retry_delay)
                return self.fetch_gpu_details(url, num_retries + 1)
            raise

        soup = BeautifulSoup(response.text, "html.parser")
        specs: Dict[str, Dict[str, Any]] = {}

        spec_sections = soup.find_all("section", {"class": "details"})
        for spec_section in spec_sections:
            title = spec_section.find("h2").get_text(strip=True)
            specs[title] = {}
            for row in spec_section.find_all("dl"):
                spec_name = row.find("dt").get_text(strip=True)
                spec_value = row.find("dd").get_text(strip=True)
                specs[title][spec_name] = spec_value

        return specs

    def fetch_gpus_by_year(
        self,
        year: int,
        courtesy_delay: float = DEFAULT_COURTESY_DELAY,
        use_tqdm: bool = False,
        desc: Optional[str] = None,
    ) -> Iterable[Tuple[str, str, Dict[str, Dict[str, Any]]]]:
        """
        Fetches the detailed specs for all GPUs released in a given year.

        >>> tpu = TechPowerUp()
        >>> iterator = tpu.fetch_gpus_by_year(2020, courtesy_delay=0) # We're only sending one request
        >>> mfr, name, specs = next(iterator)
        >>> mfr
        'NVIDIA'
        >>> name
        'A100 PCIe 40 GB'
        >>> specs["Graphics Processor"]["Architecture"]
        'Ampere'
        """
        gpu_list: List[Tuple[str, str, str]] = []
        for manufacturer in self.manufacturers:
            for gpu_name, link in self.fetch_gpu_list(
                manufacturer=manufacturer,
                year=year
            ):
                gpu_list.append((manufacturer, gpu_name, link))
            time.sleep(courtesy_delay)

        num_gpus = len(gpu_list)
        iterable: Iterable[Tuple[str, str, str]] = gpu_list

        if use_tqdm:
            try:
                from tqdm import tqdm
                iterable = tqdm(gpu_list, desc=desc)
            except ImportError:
                warnings.warn("Could not import tqdm. Falling back to non-verbose mode. Run 'pip install tqdm' to enable verbose mode.")

        for i, (mfr, gpu_name, link) in enumerate(iterable):
            specs = self.fetch_gpu_details(link)
            yield mfr, gpu_name, specs
            if i < num_gpus - 1:
                time.sleep(courtesy_delay)

    def fetch_all(
        self,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        courtesy_delay: float = DEFAULT_COURTESY_DELAY,
        use_tqdm: bool = False,
    ) -> Iterable[Tuple[str, str, Dict[str, Dict[str, Any]]]]:
        """
        Fetches the detailed specs for all GPUs released between two years.

        >>> tpu = TechPowerUp()
        >>> iterator = tpu.fetch_all(start_year=2020, courtesy_delay=0) # We're only sending one request
        >>> mfr, name, specs = next(iterator)
        >>> mfr
        'NVIDIA'
        >>> name
        'A100 PCIe 40 GB'
        >>> specs["Graphics Processor"]["Architecture"]
        'Ampere'
        """
        for year in range(start_year, end_year + 1):
            for gpu_manufacturer, gpu_name, gpu_specs in self.fetch_gpus_by_year(
                year=year,
                courtesy_delay=courtesy_delay,
                use_tqdm=use_tqdm,
                desc=f"{year}"
            ):
                yield gpu_manufacturer, gpu_name, gpu_specs

    def fetch_to_file(
        self,
        *output_files: str,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        courtesy_delay: float = DEFAULT_COURTESY_DELAY,
        use_tqdm: bool = False,
    ) -> None:
        """
        Fetches the detailed specs for all GPUs released between two years and saves them to a file.
        """
        from dbgpu.gpu import GPUSpecification
        rows: List[Dict[str, Any]] = []
        for manufacturer, name, specs in self.fetch_all(
            start_year=start_year,
            end_year=end_year,
            courtesy_delay=courtesy_delay,
            use_tqdm=use_tqdm,
        ):
            gpu_spec = GPUSpecification.from_tpu_dict(
                name=name,
                manufacturer=manufacturer, # type: ignore[arg-type]
                data=specs
            )
            rows.append(gpu_spec.to_dict())

        for output_file in output_files:
            name, ext = os.path.splitext(os.path.basename(output_file))
            if ext == ".json":
                with open(output_file, "w") as file:
                    json.dump(rows, file, default=json_serialize)
            elif ext == ".csv":
                with open(output_file, "w") as file:
                    writer = csv.writer(file)
                    writer.writerow(rows[0].keys())
                    for row in rows:
                        writer.writerow(row.values())
            elif ext == ".pkl":
                with open(output_file, "wb") as file:
                    pickle.dump(rows, file)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
