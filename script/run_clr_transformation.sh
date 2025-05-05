#!/bin/sh

cd ../CSDI_phylo
input_profile_RA="../indata/DIABIMMUNE_16S/rel-species-table.csv"

pseudo_count=$(python3 clr_transformation.py $input_profile_RA "Y")

