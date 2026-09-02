from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Summarize E0-v2 JSON runs into one CSV")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        protocol = payload["protocol"]
        source = protocol.get("source", {})
        source_name = source.get("kind", "unknown")
        source_model = source.get("model") or source.get("pretrained_spec", {}).get("architecture") or "current"
        for transition, result in payload["results"].items():
            decision = result["decision"]
            closure = result["known_control_gap_closure_fraction_n2plus"]
            linkage = result.get("downstream_linkage") or {}
            for rep_name, metrics in result["metrics"].items():
                rows.append(
                    {
                        "file": str(input_path),
                        "source_kind": source_name,
                        "source_model": source_model,
                        "transition": transition,
                        "representation": rep_name,
                        "cells": metrics.get("cells"),
                        "child_mae": metrics.get("child_mae"),
                        "parent_mae": metrics.get("parent_mae"),
                        "composition_l1": metrics.get("composition_l1"),
                        "n2p_cells": metrics.get("n2p_cells"),
                        "n2p_child_mae": metrics.get("n2p_child_mae"),
                        "n2p_parent_mae": metrics.get("n2p_parent_mae"),
                        "n2p_composition_l1": metrics.get("n2p_composition_l1"),
                        "relative_native_vs_pre_n2p": result["relative_n2plus_degradation"].get("native_vs_pre"),
                        "relative_native_vs_s2d_pca_n2p": result["relative_n2plus_degradation"].get("native_vs_s2d_pca"),
                        "s2d_pca_gap_closure": closure.get("s2d_pca_budget"),
                        "avgpool_pca_gap_closure": closure.get("avgpool_pca_budget"),
                        "blurpool_pca_gap_closure": closure.get("blurpool_pca_budget"),
                        "rho_excess_vs_abs_local_error": linkage.get("spearman_excess_vs_abs_local_count_error"),
                        "rho_excess_vs_underestimate": linkage.get("spearman_excess_vs_underestimate"),
                        "screen_go": decision.get("screen_go"),
                        "linkage_go": decision.get("linkage_go"),
                        "final_go": decision.get("final_go"),
                    }
                )
    if not rows:
        raise RuntimeError("No rows found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
