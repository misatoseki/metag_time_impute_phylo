#!/bin/sh

cd ../LOCF
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/LOCF_MR-${MISSING_RATE}.log"
python3 LOCF.py "../indata/DIABIMMUNE_16S/rel-species-table_clr.csv" "../indata/DIABIMMUNE_16S/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

