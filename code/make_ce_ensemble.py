"""Average of z-scored cross-encoder scores (zero-shot MedCPT-Cross-Encoder + our fine-tuned copy), z-scoring with the
validation-split statistics of each model for both splits. Writes cache/ce_ensemble_expanded_{val,eval}30.json."""
import json

import numpy as np

from common import CACHE

files = ["ce_ncbi_MedCPT-Cross-Encoder_expanded_{}30.json", "ce_.._cache_medcpt-ce-ft_expanded_{}30.json"]
S = {f: {sp: json.load(open(f"{CACHE}/{f.format(sp)}")) for sp in ["val", "eval"]} for f in files}
stats = {f: (np.concatenate(list(S[f]["val"].values())).mean(), np.concatenate(list(S[f]["val"].values())).std()) for f in files}
for sp in ["val", "eval"]:
    out = {k: np.mean([(np.array(S[f][sp][k]) - stats[f][0]) / stats[f][1] for f in files], 0).tolist() for k in S[files[0]][sp]}
    json.dump(out, open(f"{CACHE}/ce_ensemble_expanded_{sp}30.json", "w"))
print("written")
