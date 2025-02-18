#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DIR=$(realpath ${SCRIPT_DIR}/..)
PYTHONPATH=$DIR/src/:$PYTHONPATH
export PYTHONPATH
python -m dbgpu "$@"
