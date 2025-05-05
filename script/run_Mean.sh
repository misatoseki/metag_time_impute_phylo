#!/bin/sh

cd ../Mean
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/DIABIMMUNE_16S/rel-species-table_clr.csv" "../indata/DIABIMMUNE_16S/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

