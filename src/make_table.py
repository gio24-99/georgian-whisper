
import glob
import json
import os

import pandas as pd

RESULTS_DIR = "results"


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_metrics.json"))):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "name": m["name"],
            "model": m["model"] + (f" + {m['adapter']}" if m.get("adapter") else ""),
            "split": m["split"],
            "clips": m["num_samples"],
            "WER %": m["wer"],
            "CER %": m["cer"],
            "real-time factor": m["real_time_factor"],
            "device": m["device"],
        })
    if not rows:
        raise SystemExit(f"No *_metrics.json files found in {RESULTS_DIR}/")

    df = pd.DataFrame(rows).sort_values("WER %", ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False, encoding="utf-8-sig")

    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(df.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/summary.csv and {RESULTS_DIR}/summary.md")


if __name__ == "__main__":
    main()