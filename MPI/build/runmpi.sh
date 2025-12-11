#!/usr/bin/env bash
set -euo pipefail
BIN=./lsm_price

NPROCS_LIST="1 2 4 8 16"

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
echo "nprocs,paths,S0,K,type,price,time_ms" > "$OUTFILE"

for np in $NPROCS_LIST; do
  for paths in $PATHS_LIST; do
    for case in "${CASES[@]}"; do
      read -r S0 K opttype <<< "$case"

      if [ "$opttype" = "put" ]; then
        EXTRA="--put"
      else
        EXTRA="--call"
      fi

      echo "Running: np=$np paths=$paths S0=$S0 K=$K type=$opttype"

      out=$(mpirun -np "$np" $BIN \
        --S0 "$S0" \
        --K "$K" \
        --r "$R" \
        --sigma "$SIGMA" \
        --T "$T" \
        --paths "$paths" \
        --steps "$STEPS" \
        $EXTRA)
        line=$(echo "$out" | grep "CPU price")

        price=$(echo "$line" | sed -E 's/.*CPU price = ([0-9.]+), time = ([0-9.]+) ms.*/\1/')
        time_ms=$(echo "$line" | sed -E 's/.*CPU price = ([0-9.]+), time = ([0-9.]+) ms.*/\2/')

      if [ -z "$price" ] || [ -z "$time_ms" ]; then
        echo "ERROR: Failed to parse output!"
        echo "$out"
        exit 1
      fi

      echo "$np,$paths,$S0,$K,$opttype,$price,$time_ms" >> "$OUTFILE"
    done
  done
done

echo "All runs complete and results saved to $OUTFILE"
