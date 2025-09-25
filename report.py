# atia/report.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="parsed")
    ap.add_argument("--out", default="parsed/rollups")
    args = ap.parse_args()

    p = Path(args.parsed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    samples = pd.read_json(p/"sample_rows.json") if (p/"sample_rows.json").exists() else pd.read_csv(p/"sample_rows.csv")
    evals   = pd.read_json(p/"eval_summary.json") if (p/"eval_summary.json").exists() else pd.read_csv(p/"eval_summary.csv")

    # Overall by attack
    by_attack = samples.groupby("attack_type").agg(
        n=("sample_id", "count"),
        accuracy=("score_bool", "mean")
    ).reset_index()
    by_attack.to_csv(out/"by_attack.csv", index=False)

    # By modality
    by_mod = samples.groupby("modality").agg(
        n=("sample_id", "count"),
        accuracy=("score_bool", "mean")
    ).reset_index()
    by_mod.to_csv(out/"by_modality.csv", index=False)

    # By model × attack
    # explode models (comma-separated) to long form
    evals_long = evals.copy()
    evals_long["model"] = evals_long["models"].fillna("").str.split(",").apply(lambda xs: [x.strip() for x in xs if x.strip()]) 
    evals_long = evals_long.explode("model", ignore_index=True)
    joined = samples.merge(evals_long[["eval_id","model"]], on="eval_id", how="left")

    by_model_attack = joined.groupby(["model","attack_type"]).agg(
        n=("sample_id","count"),
        accuracy=("score_bool","mean")
    ).reset_index()
    by_model_attack.to_csv(out/"by_model_attack.csv", index=False)

    print(f"[done] Wrote:\n  {out/'by_attack.csv'}\n  {out/'by_modality.csv'}\n  {out/'by_model_attack.csv'}")

if __name__ == "__main__":
    main()


