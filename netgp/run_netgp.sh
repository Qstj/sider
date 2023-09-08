#!/bin/bash


WORKDIR="./"

python netgp.py \
    --datadir $WORKDIR/Data \
    --target_fname target_onehot_example.tsv \
    --outdir $WORKDIR/Results \
    --combined_score_cutoff 800 \
    --restart_prob 0.2 \
    --out_fname drug_target_profile.out \
    --ppi_fname 9606.protein.links.symbols.v11.5.txt
