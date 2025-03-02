#!/bin/sh

#$ -cwd

cd /path/to/dir
input_profile_RA="../indata/rel-species-table.csv"

pseudo_count=$(python3 clr_transformation.py $input_profile_RA "Y")

