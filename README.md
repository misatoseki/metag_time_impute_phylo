# Diffusion model for imputing time-series gut microbiome profiles

This repository stores codes and datasets used in the paper titled "Diffusion model approach for imputing time-series gut microbiome profiles from 16S rRNA and metagenomic sequencing data".

## Repository structure
```
.
├── CSDI_phylo/ # Diffusion-based phylum-level CNN-enhanced imputation
├── CSDI_phylo_meta # Diffusion-based phylum-level CNN-enhanced imputation incorporating metadata information
├── Linear/ # Linear interpolation
├── LOCF/ # LOCF interpolation
├── Mean/ # Mean interpolation
├── indata/ # Example input files
├── outdata/ # For storing output files
├── scripts/ # Optional batch scripts
├── logs/ # For storing log files
├── requirements.txt # Python dependencies
└── README.md
```
## How to use

### Prepare input data
Prepare time-series metagenomic relative abundance profile. See `indata/DIABIMMUNE_16S/rel-species-table.csv` for reference.

### CLR transformation
If profile transformed by the centered log ratio (clr) is not avabilable, transform relative abundances by running `clr_transformation.py`. See script for reference.

### Train and Evaluate model
Run `exe.py`. See scripts for reference.

## Acknowledgement

The codes of the diffusion model were developed based on [CSDI](https://github.com/ermongroup/CSDI).
