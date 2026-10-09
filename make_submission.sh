#!/bin/bash
# Build <rollno>_nlp_bootcamp.zip with exactly the required contents: code/, predictions/, results.json, report.md.
# Usage: ./make_submission.sh <rollno>   (also writes the roll number into results.json if it is still a placeholder)
set -e
cd "$(dirname "$0")"
ROLL=${1:-$(python3 -c "import json;print(json.load(open('results.json'))['roll_no'])")}
python3 - "$ROLL" <<'PY'
import json, sys
r = json.load(open("results.json"))
if r.get("roll_no") in ("", "ROLLNO"):
    r["roll_no"] = sys.argv[1]
    json.dump(r, open("results.json", "w"), indent=2)
PY
mkdir -p submission
OUT="submission/${ROLL}_nlp_bootcamp.zip"
rm -f "$OUT"
# code/experiments/ (exploration scripts) and code/tools/ stay in the repo but are not part of the submission
zip -qr "$OUT" code predictions results.json report.md -x "code/__pycache__/*" "code/experiments/*" "code/tools/*" "code/*/__pycache__/*"
python3 check_format.py "$OUT"
echo "wrote $OUT"
