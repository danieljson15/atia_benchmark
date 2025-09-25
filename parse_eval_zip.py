# parse_eval_zip.py
#
# Usage:
#   python parse_eval_zip.py --logs "logs" --out "parsed"
#
# Reads .eval (ZIP) logs directly (no SDK), normalizes to:
#   parsed/eval_summary.{csv,json}
#   parsed/sample_rows.{csv,json}
# Adds:
#   - primary_metric_name / primary_metric_value (from sample["scores"])
#   - score_bool (derived via taxonomy.yaml -> score_map or numeric fallback)

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import zipfile

import pandas as pd
import yaml

# ------------------------ taxonomy & helpers ---------------------------------
def load_taxonomy(path="taxonomy.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    except FileNotFoundError:
        y = {}
    return (
        y.get("attack_keywords", {}),
        y.get("modality_keywords", {}),
        {k.upper(): v for k, v in (y.get("score_map", {}) or {}).items()},
    )

ATTACK_KEYWORDS, MODALITY_KEYWORDS, SCORE_MAP = load_taxonomy()

# priority order for picking a primary metric from sample["scores"]
PRIMARY_SCORE_PRIORITY = ["accuracy", "harmful_tool_invoked"]

def infer_from_keywords(s: str, table: Dict[str, str], default: Optional[str] = None) -> Optional[str]:
    s_low = (s or "").lower()
    for k, v in table.items():
        if k in s_low:
            return v
    return default

def safe_get(obj: Any, path: Iterable[str], default=None):
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return cur if cur is not None else default

def first_nonempty(*vals):
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

def read_json_from_zip(z: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    with z.open(member) as f:
        return json.loads(f.read().decode("utf-8"))

# ------------------------ scoring normalization ------------------------------
def extract_primary_metric(scores: Any) -> Tuple[Optional[str], Optional[Any]]:
    """
    scores is typically a dict like: {"metric_name": {"value": "C", ...}, ...}
    Returns (primary_metric_name, primary_metric_value)
    """
    if not isinstance(scores, dict):
        return None, None
    # priority match
    for k in PRIMARY_SCORE_PRIORITY:
        v = scores.get(k)
        if isinstance(v, dict) and "value" in v:
            return k, v["value"]
    # first key with .value
    for k, v in scores.items():
        if isinstance(v, dict) and "value" in v:
            return k, v["value"]
    return None, None

def coerce_score_bool(primary_value: Any, raw_score: Any) -> Optional[float]:
    """
    Try to derive a [0,1] value:
      1) Map primary_value via SCORE_MAP (string labels)
      2) Else, if raw_score is numeric, clip to [0,1]
      3) Else, if raw_score is string, map via SCORE_MAP
      4) Else, return None
    """
    # 1) label from primary metric
    if isinstance(primary_value, (str, int, float)):
        key = str(primary_value).strip().upper()
        if key in SCORE_MAP:
            return float(SCORE_MAP[key])

    # 2) numeric raw_score
    if isinstance(raw_score, (int, float)):
        x = float(raw_score)
        if x < 0: x = 0.0
        if x > 1: x = 1.0
        return x

    # 3) string raw_score
    if isinstance(raw_score, str):
        key = raw_score.strip().upper()
        if key in SCORE_MAP:
            return float(SCORE_MAP[key])

    return None

# ------------------------ main parsing ---------------------------------------
def parse_one_eval(eval_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with zipfile.ZipFile(eval_path, "r") as z:
        names = set(z.namelist())
        if "_journal/start.json" not in names:
            raise RuntimeError("Missing _journal/start.json")

        start = read_json_from_zip(z, "_journal/start.json")
        eval_info = start.get("eval", {})    # eval_id, task, task_id, created, task_attribs, ...
        plan = start.get("plan", {})         # sometimes has models/model

        # eval-level fields
        eval_id = first_nonempty(eval_info.get("eval_id"), eval_path.stem)
        task = first_nonempty(eval_info.get("task"), eval_info.get("task_registry_name"))
        task_id = eval_info.get("task_id")
        created = eval_info.get("created")
        category = safe_get(eval_info, ["task_attribs", "category"])
        models = plan.get("models") or plan.get("model") or []
        if isinstance(models, str): models = [models]
        models = [m for m in models if m]

        # infer attack/modality
        attack_type = first_nonempty(
            category,
            infer_from_keywords(task or "", ATTACK_KEYWORDS),
            infer_from_keywords(task_id or "", ATTACK_KEYWORDS),
        )
        modality = first_nonempty(
            infer_from_keywords(task or "", MODALITY_KEYWORDS),
            "text",
        )

        # sample files
        sample_members = sorted([n for n in names if n.startswith("samples/") and n.endswith(".json")])

        sample_rows: List[Dict[str, Any]] = []
        for m in sample_members:
            s = read_json_from_zip(z, m)

            sid = first_nonempty(s.get("id"), s.get("sample_id"), s.get("name"))
            # scores container and any raw numeric/string score field
            scores = s.get("scores") or {}
            raw_score = first_nonempty(s.get("score"), safe_get(s, ["result", "score"]))

            primary_name, primary_value = extract_primary_metric(scores)
            score_bool = coerce_score_bool(primary_value, raw_score)

            smeta: Dict[str, Any] = first_nonempty(s.get("meta"), s.get("metadata"), {}) or {}
            stags = first_nonempty(s.get("tags"), smeta.get("tags"), []) or []

            s_attack = first_nonempty(
                smeta.get("attack_type"),
                infer_from_keywords(" ".join(map(str, stags)), ATTACK_KEYWORDS),
                attack_type,
            )
            s_modality = first_nonempty(
                smeta.get("modality"),
                infer_from_keywords(" ".join(map(str, stags)), MODALITY_KEYWORDS),
                modality,
            )

            sample_rows.append({
                "eval_id": eval_id,
                "eval_file": eval_path.name,
                "sample_id": sid,
                "attack_type": s_attack,
                "modality": s_modality,
                # scoring
                "primary_metric_name": primary_name,
                "primary_metric_value": primary_value,
                "score": raw_score,
                "score_bool": score_bool,
                # misc
                "tags": stags,
            })

        eval_row = {
            "eval_id": eval_id,
            "eval_file": eval_path.name,
            "task": task,
            "task_id": task_id,
            "attack_type": attack_type,
            "modality": modality,
            "models": ", ".join(models) if models else None,
            "created": created,
            "num_samples": len(sample_rows),
        }

        return eval_row, sample_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="Path to .eval file or directory with .eval files")
    ap.add_argument("--out", default="parsed", help="Output directory")
    args = ap.parse_args()

    logs_path = Path(args.logs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect .evals
    if logs_path.is_file() and logs_path.suffix.lower() == ".eval":
        eval_paths = [logs_path]
    else:
        eval_paths = sorted(logs_path.rglob("*.eval"))

    if not eval_paths:
        print(f"[warn] No .eval files found under: {logs_path}")
        return

    eval_rows: List[Dict[str, Any]] = []
    all_samples: List[Dict[str, Any]] = []

    for p in eval_paths:
        try:
            e_row, s_rows = parse_one_eval(p)
            eval_rows.append(e_row)
            all_samples.extend(s_rows)
            print(f"[ok] {p.name}: {e_row['num_samples']} samples")
        except Exception as e:
            print(f"[fail] {p.name}: {e}")

    # export
    eval_df = pd.DataFrame(eval_rows)
    samples_df = pd.DataFrame(all_samples)

    (out_dir / "rollups").mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(out_dir / "eval_summary.csv", index=False)
    samples_df.to_csv(out_dir / "sample_rows.csv", index=False)
    eval_df.to_json(out_dir / "eval_summary.json", orient="records", indent=2)
    samples_df.to_json(out_dir / "sample_rows.json", orient="records", indent=2)

    print(f"[done] Wrote:\n  {out_dir/'eval_summary.csv'}\n  {out_dir/'sample_rows.csv'}\n  {out_dir/'eval_summary.json'}\n  {out_dir/'sample_rows.json'}")

if __name__ == "__main__":
    main()