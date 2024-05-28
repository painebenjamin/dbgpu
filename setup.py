import os
import re
import sys

from setuptools import find_packages, setup

deps = ["click", "pydantic"]
build_deps = ["requests", "beautifulsoup4", "tqdm"]

extra_deps = {
    "socks": ["requests[socks]"],
    "tabulate": ["tabulate"],
    "fuzz": ["thefuzz"],
    "build": build_deps,
    "all": ["requests[socks]", "tabulate", "thefuzz"] + build_deps,
}

setup(
    name="dbgpu",
    description="A small, easy-to-use open source database of over 2000 GPUs with architecture, manufacturing, API support and performance details.",
    version=open("src/dbgpu/version.txt", "r", encoding="utf-8").read().strip(),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    url="https://github.com/painebenjamin/dbgpu",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"dbgpu": ["py.typed", "version.txt", "data.pkl"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=deps,
    extras_require=extra_deps,
    entry_points={
        "console_scripts": [
            "dbgpu = dbgpu.__main__:main"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, 13)],
)
