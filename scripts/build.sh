#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DIR=$(realpath ${SCRIPT_DIR}/..)

# Remove old build if present
rm -rf ${DIR}/dist ${DIR}/src/dbgpu.egg-info

# Write the version file
date +"%Y.%m" > ${DIR}/src/dbgpu/version.txt

# Build the package
cd $DIR
python setup.py sdist

# Remove build artifacts
rm -rf ${DIR}/src/dbgpu.egg-info
