#!/usr/bin/env bash
set -euo pipefail

# Adjust this if your executable is named differently (e.g. ./lsm_price)
BIN=./lsm_gpu

R=0.05
SIGMA=0.2
T=1.0
STEPS=50

PATHS_LIST="50000 200000 500000 1000000"

CASES=(
  "100 100 call"
  "100 100 put"
  "100 90 call"
  "100 90 put"
  "90 100 call"
  "90 100 put"
)

OUTFILE="results.csv"
echo "paths,S0,K,type,price,time_ms" > "$OUTFILE"

for paths in $PATHS_LIST; do
  for case in "${CASES[@]}"; do
    read -r S0 K opttype <<< "$case"

    if [ "$opttype" = "put" ]; then
      EXTRA="--put"
    else
      EXTRA="--call"
    fi

    echo "Running: paths=$paths S0=$S0 K=$K type=$opttype"

    out=$($BIN \
      --S0 "$S0" \
      --K "$K" \
      --r "$R" \
      --sigma "$SIGMA" \
      --T "$T" \
      --paths "$paths" \
      --steps "$STEPS" \
      $EXTRA)


    price=$(echo "$out" | sed -nE 's/.*Estimated American option price: ([0-9.eE+-]+).*/\1/p')

    time_ms=$(echo "$out" | sed -nE 's/.*Total GPU time[^(]*\(ms\):[[:space:]]*([0-9.eE+-]+).*/\1/p')

    if [ -z "$price" ] || [ -z "$time_ms" ]; then
      echo "ERROR: failed to parse output for paths=$paths S0=$S0 K=$K type=$opttype" >&2
      echo "$out" >&2
      exit 1
    fi

    echo "$paths,$S0,$K,$opttype,$price,$time_ms" >> "$OUTFILE"
  done
done

echo "All runs complete and results saved to $OUTFILE"
