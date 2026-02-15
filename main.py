#!/usr/bin/env python3
"""DocDrift CLI — Detect cognitive drift between docs and code."""
import argparse
import json
import sys
from docdrift import analyze


def main():
    """Entry point for the DocDrift CLI."""
    parser = argparse.ArgumentParser(description="DocDrift: detect doc-code cognitive drift")
    parser.add_argument("files", nargs="+", help="Python files to analyze")
    parser.add_argument("--threshold", type=float, default=0.7, help="Drift threshold 0-1 (default: 0.7)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit code 1 if drift detected")
    args = parser.parse_args()

    all_reports = {}
    has_drift = False

    for filepath in args.files:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        reports = analyze(source, threshold=args.threshold)
        all_reports[filepath] = reports
        if any(r["status"] in ("drifted", "missing_docs") for r in reports):
            has_drift = True

    if args.as_json:
        print(json.dumps(all_reports, indent=2))
    else:
        for filepath, reports in all_reports.items():
            print(f"\n\U0001f4c4 {filepath}")
            print("\u2500" * 50)
            if not reports:
                print("  No functions or classes found.")
            for r in reports:
                icons = {"drifted": "\U0001f534", "missing_docs": "\U0001f7e1", "aligned": "\U0001f7e2"}
                print(f"  {icons[r['status']]} {r['message']}")
        drifted_count = sum(1 for rs in all_reports.values() for r in rs if r["status"] != "aligned")
        total_count = sum(len(rs) for rs in all_reports.values())
        print(f"\nSummary: {drifted_count}/{total_count} elements with drift issues")

    if args.fail_on_drift and has_drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
