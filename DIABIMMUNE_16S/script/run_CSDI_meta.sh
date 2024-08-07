#!/bin/bash

#$ -cwd

cd /path/to/dir
LOGPATH=../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/CSDI_meta_MR-${MISSING_RATE}.log"
python3 exe.py \
    --testmissingratio $MISSING_RATE \
    --batch_size 8 \
    --epochs 3000 \
    --metalist "allergy_milk" "allergy_egg" "allergy_peanut" | tee $LOG

