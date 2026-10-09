#!/bin/bash
# usage: code/tools/run_bg.sh <logname> <script.py> [args...]
# Runs a pipeline script from code/, writes a filtered log to cache/<logname>.log ending with "EXIT <code>".
# Used for long CPU jobs started with `setsid nohup ... &` so they survive the terminal.
cd "$(dirname "$0")/.." || exit 1
log=../cache/$1.log; shift
TRANSFORMERS_VERBOSITY=error python3 "$@" > "$log.raw" 2>&1; code=$?
grep -vE "Warn|warn|it/s\]" "$log.raw" > "$log"; echo "EXIT $code" >> "$log"
