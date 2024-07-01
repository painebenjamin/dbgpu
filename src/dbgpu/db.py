from __future__ import annotations

import os
import json
import csv
import pickle

from typing import Dict, List, TYPE_CHECKING

from dbgpu.constants import DEFAULT_GPU_DB_PATH, DEFAULT_GPU_DB
from dbgpu.util import safe_name

if TYPE_CHECKING:
    from pandas import DataFrame
    from dbgpu.gpu import GPUSpecification

__all__ = ["GPUDatabase"]

class GPUDatabase:
    """
    A class to look up GPU specifications.
    """
    specifications: Dict[str, GPUSpecification]
    manufacturer_prefixed_name_map: Dict[str, str]

    def __init__(self, specifications: List[GPUSpecification] = []) -> None:
        self.specifications = {}
        self.manufacturer_prefixed_name_map = {}
        for spec in specifications:
            self.specifications[spec.name_key] = spec
            self.manufacturer_prefixed_name_map[spec.manufacturer_prefixed_name_key] = spec.name_key

    @classmethod
    def from_file(cls, path: str) -> GPUDatabase:
        """
        Loads a GPUDatabase from a JSON or CSV file.
        """
        from dbgpu.gpu import GPUSpecification
        basename, ext = os.path.splitext(os.path.basename(path))
        specs: List[GPUSpecification] = []
        if ext == ".json":
            with open(path, "r") as file:
                data = json.load(file)
            for gpu in data:
                gpu = {k.strip(): v if v != "" else None for k, v in gpu.items()}
                specs.append(GPUSpecification(**gpu))
        elif ext == ".csv":
            with open(path, "r") as file:
                reader = csv.DictReader(file)
                for gpu in reader:
                    gpu = {k.strip(): v if v != "" else None for k, v in gpu.items()}
                    specs.append(GPUSpecification(**gpu))
        elif ext == ".pkl":
            specs.extend([
                GPUSpecification(**gpu)
                for gpu in pickle.load(open(path, "rb"))
            ])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return cls(specs)

    @classmethod
    def default(cls) -> GPUDatabase:
        """
        Returns a default GPUDatabase with the built-in GPU specifications.
        """
        global DEFAULT_GPU_DB
        if DEFAULT_GPU_DB is None:
            DEFAULT_GPU_DB = cls.from_file(DEFAULT_GPU_DB_PATH) # type: ignore[assignment]
        return DEFAULT_GPU_DB # type: ignore[return-value]

    @property
    def dataframe(self) -> DataFrame:
        """
        Returns the GPU database as a pandas DataFrame.
        """
        if not hasattr(self, "_dataframe"):
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas is required to convert the GPU database to a DataFrame. Run `pip install pandas` to install it.")
            self._dataframe = pd.DataFrame([
                gpu.to_dict() for gpu in self.specifications.values()
            ])
        return self._dataframe

    @property
    def names(self) -> List[str]:
        """
        Returns a list of the names of all GPU specifications in the database.
        """
        if not hasattr(self, "_names"):
            self._names = [
                spec.name
                for spec in self.specifications.values()
            ]
        return self._names

    @property
    def specs(self) -> List[GPUSpecification]:
        """
        Returns a list of all GPU specifications in the database.
        """
        return list(self.specifications.values())

    def search(self, name: str, min_score: int=75) -> GPUSpecification:
        """
        Uses fuzzy matching to find the GPU specification with the given name.
        """
        try:
            from thefuzz import process # type: ignore[import]
        except ImportError:
            try:
                from fuzzywuzzy import process # type: ignore[import]
            except ImportError:
                raise ImportError("thefuzz or fuzzywuzzy is required to search for GPU specifications. Run `pip install thefuzz` to install it.")
        [(name, score)] = process.extract(name, self.names, limit=1)
        if score < min_score:
            raise KeyError(f"GPU specification with name '{name}' not found.")
        return self[name]

    def __getitem__(self, key: str) -> GPUSpecification:
        """
        Returns the GPU specification with the given name.
        """
        key = safe_name(key)
        if key in self.manufacturer_prefixed_name_map:
            key = self.manufacturer_prefixed_name_map[key]
        return self.specifications[key]
