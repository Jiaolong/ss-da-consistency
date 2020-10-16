#! /usr/bin/env bash
set -e
set -x
for exp in art_painting cartoon sketch photo; do
    for s in 100 200 300; do python3 main.py --config config/pacs/${exp}/uda.yaml --seed $s; done
done
