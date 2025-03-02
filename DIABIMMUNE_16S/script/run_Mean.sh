#!/bin/sh

#$ -cwd
#$ -l v100=1,s_vmem=100G

cd /path/to/dir
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.2"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.3"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.4"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.5"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.6"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.7"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.8"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG

MISSING_RATE="0.9"

LOG="${LOGPATH}/Mean_MR-${MISSING_RATE}.log"
python3 Mean.py "../indata/rel-species-table_clr.csv" "../indata/mask_${MISSING_RATE}.csv" ${MISSING_RATE} | tee $LOG


