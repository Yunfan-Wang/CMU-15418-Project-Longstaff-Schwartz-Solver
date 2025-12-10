#!/usr/bin/env python3
import csv
import math
import argparse
from collections import defaultdict, namedtuple

Row = namedtuple("Row", ["paths", "S0", "K", "opt_type", "price", "time_ms"])


def read_results_csv(path):
    data = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            try:
                paths = int(line["paths"])
                S0 = float(line["S0"])
                K = float(line["K"])
                opt_type = line["type"].strip()
                price = float(line["price"])
                time_ms = float(line["time_ms"])
            except KeyError as e:
                raise RuntimeError(f"Missing expected column in {path}: {e}")

            key = (paths, S0, K, opt_type)
            data[key] = Row(paths, S0, K, opt_type, price, time_ms)
    return data


def summarize_by_pathsize(base_data, curr_data):

    per_paths = defaultdict(lambda: {
        "speedups": [],
        "price_errors": [],
        "rel_errors": [],
    })

    keys_intersection = set(base_data.keys()) & set(curr_data.keys())
    if not keys_intersection:
        raise RuntimeError("No overlapping (paths,S0,K,type) rows")

    for key in sorted(keys_intersection):
        base_row = base_data[key]
        curr_row = curr_data[key]

        assert base_row.paths == curr_row.paths
        paths = base_row.paths

        if curr_row.time_ms <= 0:
            continue 
        speedup = base_row.time_ms / curr_row.time_ms

        price_err = curr_row.price - base_row.price
        if abs(base_row.price) > 1e-12:
            rel_err = abs(price_err) / abs(base_row.price)
        else:
            rel_err = 0.0 

        bucket = per_paths[paths]
        bucket["speedups"].append(speedup)
        bucket["price_errors"].append(price_err)
        bucket["rel_errors"].append(rel_err)

    summary = {}
    for paths, bucket in sorted(per_paths.items()):
        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        def var(xs):
            if len(xs) < 2:
                return float("nan")
            m = mean(xs)
            return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

        speedups = bucket["speedups"]
        price_errors = bucket["price_errors"]
        rel_errors = bucket["rel_errors"]

        avg_speedup = mean(speedups)
        var_speedup = var(speedups)
        std_speedup = math.sqrt(var_speedup) if not math.isnan(var_speedup) else float("nan")

        avg_price_err = mean(price_errors)
        var_price_err = var(price_errors)
        rmse_price_err = math.sqrt(mean([e**2 for e in price_errors])) if price_errors else float("nan")

        avg_rel_err = mean(rel_errors)

        summary[paths] = {
            "n_cases": len(speedups),
            "avg_speedup": avg_speedup,
            "std_speedup": std_speedup,
            "avg_price_error": avg_price_err,
            "var_price_error": var_price_err,
            "rmse_price_error": rmse_price_err,
            "mean_rel_error": avg_rel_err,
        }

    return summary


def print_summary(summary):
    print("Per-pathsize summary (baseline vs current model):")
    print()
    header = (
        "paths",
        "n_cases",
        "avg_speedup",
        "std_speedup",
        "avg_price_error",
        "var_price_error",
        "rmse_price_error",
        "mean_rel_error",
    )
    print(",".join(header))

    for paths in sorted(summary.keys()):
        s = summary[paths]
        row = [
            str(paths),
            str(s["n_cases"]),
            f"{s['avg_speedup']:.4f}",
            f"{s['std_speedup']:.4f}" if not math.isnan(s["std_speedup"]) else "nan",
            f"{s['avg_price_error']:.6f}" if not math.isnan(s["avg_price_error"]) else "nan",
            f"{s['var_price_error']:.6f}" if not math.isnan(s["var_price_error"]) else "nan",
            f"{s['rmse_price_error']:.6f}" if not math.isnan(s["rmse_price_error"]) else "nan",
            f"{s['mean_rel_error']:.6f}" if not math.isnan(s["mean_rel_error"]) else "nan",
        ]
        print(",".join(row))


def write_summary_csv(summary, out_path):
    fieldnames = [
        "paths",
        "n_cases",
        "avg_speedup",
        "std_speedup",
        "avg_price_error",
        "var_price_error",
        "rmse_price_error",
        "mean_rel_error",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paths in sorted(summary.keys()):
            s = summary[paths]
            writer.writerow({
                "paths": paths,
                "n_cases": s["n_cases"],
                "avg_speedup": s["avg_speedup"],
                "std_speedup": s["std_speedup"],
                "avg_price_error": s["avg_price_error"],
                "var_price_error": s["var_price_error"],
                "rmse_price_error": s["rmse_price_error"],
                "mean_rel_error": s["mean_rel_error"],
            })


def main():
    parser = argparse.ArgumentParser(
        description="parser"
    )
    parser.add_argument("baseline_csv")
    parser.add_argument("current_csv")
    parser.add_argument("--out")
    args = parser.parse_args()

    base_data = read_results_csv(args.baseline_csv)
    curr_data = read_results_csv(args.current_csv)

    summary = summarize_by_pathsize(base_data, curr_data)
    print_summary(summary)

    if args.out:
        write_summary_csv(summary, args.out)
        print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()