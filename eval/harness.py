"""Eval harness for the Company Brain.

Three metrics, one of which is a hard gate:

  - faithfulness          : LLM judge — does the answer follow from cited chunks?
  - cross_source_synthesis: did the answer cite ≥2 different sources when expected?
  - acl_leak_rate         : HARD GATE — any private doc surfacing to an
                            unauthorized user fails the run, regardless of quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from brain.pipeline import answer_for, DEFAULT_GROUPS


def evaluate(eval_set_path: str, config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    eval_set = json.loads(Path(eval_set_path).read_text())

    faithful = 0
    crosssrc_ok = 0
    crosssrc_expected = 0
    leak_count = 0
    n = 0

    for case in eval_set["cases"]:
        n += 1
        a = answer_for(case["user"], case["query"], cfg, groups=DEFAULT_GROUPS)

        # 1. Faithfulness — cheap proxy: every must_contain phrase is present
        if all(needle.lower() in a.answer.lower() for needle in case.get("must_contain", [])):
            faithful += 1

        # 2. Cross-source synthesis — were ≥2 distinct sources cited?
        if case.get("expects_cross_source"):
            crosssrc_expected += 1
            sources = {cid.split("::")[0] for cid in a.citations}
            if len(sources) >= 2:
                crosssrc_ok += 1

        # 3. ACL leak — did any forbidden chunk appear?
        forbidden = set(case.get("forbidden_chunk_ids", []))
        if forbidden & set(a.citations):
            leak_count += 1
            print(f"[LEAK] {case['id']} user={case['user']} leaked={sorted(forbidden & set(a.citations))}",
                  file=sys.stderr)

    report = {
        "n": n,
        "faithfulness": round(faithful / n, 3) if n else 0.0,
        "cross_source_synthesis": (
            round(crosssrc_ok / crosssrc_expected, 3) if crosssrc_expected else None
        ),
        "acl_leak_rate": round(leak_count / n, 3) if n else 0.0,
        "acl_leaks": leak_count,
        "gate_pass": leak_count == 0,
    }
    print(json.dumps(report, indent=2))
    if not report["gate_pass"]:
        sys.exit(2)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    evaluate(args.eval_set, args.config)


if __name__ == "__main__":
    main()
