#!/bin/bash
# usage: run_bg.sh <logname> <python args...>  -- runs from code/, filtered log in cache/<logname>.log, ends with EXIT <code>
log=../cache/$1.log; shift
TRANSFORMERS_VERBOSITY=error python3 "$@" > $log.raw 2>&1; code=$?
grep -vE "Warn|warn|it/s\]" $log.raw > $log; echo "EXIT $code" >> $log
