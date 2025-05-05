#!/bin/sh

cd ../CSDI_phylo
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/CSDI_phylo_MR-${MISSING_RATE}.log"
python3 exe.py \
    --testmissingratio $MISSING_RATE \
    --batch_size 8 \
    --epochs 3000 \
    --dataset_path "../indata/DIABIMMUNE_16S/rel-species-table_clr.csv" \
    --mask_path "../indata/DIABIMMUNE_16S/mask_${MISSING_RATE}.csv" | tee $LOG

