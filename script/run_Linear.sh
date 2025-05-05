#!/bin/sh

cd ../Linear
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/Linear_MR-${MISSING_RATE}.log"
python3 Linear.py "../indata/DIABIMMUNE_16S/rel-species-table_clr.csv" "../indata/DIABIMMUNE_16S/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

