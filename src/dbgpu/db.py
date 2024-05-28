from __future__ import annotations

import os
import json
import csv
import pickle

from typing import Dict, List, TYPE_CHECKING

from dbgpu.constants import DEFAULT_GPU_DB_PATH, DEFAULT_GPU_DB

if TYPE_CHECKING:
    from pandas import DataFrame
    from dbgpu.gpu import GPUSpecification

__all__ = ["GPUDatabase"]

class GPUDatabase:
    """
    A class to look up GPU specifications.
    """
    specifications: Dict[str, GPUSpecification]

    def __init__(self, specifications: List[GPUSpecification] = []) -> None:
        self.specifications = {}
        for spec in specifications:
            self.specifications[spec.name] = spec

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

    def __getitem__(self, key: str) -> GPUSpecification:
        """
        Returns the GPU specification with the given name.
        """
        return self.specifications[key]
