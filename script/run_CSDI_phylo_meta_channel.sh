#!/bin/sh

cd ../CSDI_phylo_meta_channel
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/CSDI_phylo_meta_channel_MR-${MISSING_RATE}.log"
python3 exe.py \
    --testmissingratio $MISSING_RATE \
    --batch_size 8 \
    --epochs 3000 \
    --dataset_path "../indata/DIABIMMUNE_16S/rel-species-table_clr_country.csv" \
    --num_cat 1 | tee $LOG

