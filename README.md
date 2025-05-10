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
Prepare two input files.
+ Time-series metagenomic relative abundance profile. See `indata/DIABIMMUNE_16S/rel-species-table.csv` for reference.
+ Mask data. See `indata/DIABIMMUNE_16S/mask_0.1.csv` for reference.

### CLR transformation
If profile transformed by the centered log ratio (clr) is not avabilable, transform relative abundances by running `clr_transformation.py`. 
```
cd CSDI_phylo
input_profile_RA="../indata/DIABIMMUNE_16S/rel-species-table.csv" # Replace as appropriate
python clr_transformation.py $input_profile_RA "Y"
```
See script `run_clr_transformation.sh` for reference.

### Train and Evaluate model
Run `exe.py`. 
```
input_profile_CLR="../indata/DIABIMMUNE_16S/rel-species-table_clr.csv" # Replace as appropriate
input_mask_data="../indata/DIABIMMUNE_16S/mask_0.1.csv" # Replace as appropriate
python exe.py --testmissingratio 0.1 --dataset_path $input_profile_CLR --mask_path $input_mask_data
```
See script `run_CSDI_phylo.sh` for reference.

## Acknowledgement

The codes of the diffusion model were developed based on [CSDI](https://github.com/ermongroup/CSDI).
