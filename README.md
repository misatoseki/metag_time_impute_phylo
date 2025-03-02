# Diffusion model for imputing time-series gut microbiome profiles

This repository stores codes and datasets used in the paper titled "Diffusion model approach for imputing time-series gut microbiome profiles from 16S rRNA and metagenomic sequencing data".

## Pre-requisites

+ python >=3.12
+ pickle
+ numpy
+ pandas
+ torch
+ argparse
+ datetime
+ json
+ yaml
+ scikit-learn
+ tqdm
+ category_encoders

## How to use

### Prepare input data
Prepare time-series metagenomic relative abundance profile. See `indata/rel-species-table.csv` for reference.

### CLR transformation
Transform relative abundances using the centered log ratio (clr) by running `clr_transformation.py`. See script for reference.

### Train and Evaluate model
Run `exe.py`. See scripts for reference.

## Acknowledgement

The codes of the diffusion model were developed based on [CSDI](https://github.com/ermongroup/CSDI).
