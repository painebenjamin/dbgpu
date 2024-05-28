#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

${SCRIPT_DIR}/run-type-check.sh
${SCRIPT_DIR}/run-import-check.sh
${SCRIPT_DIR}/run-unit-test.sh
