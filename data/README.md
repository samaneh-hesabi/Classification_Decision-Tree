<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Data Directory</div>

This directory contains all datasets used in the project, including raw data, processed data, and intermediate data files.

# 1. Contents
- `raw/`: Original, unprocessed datasets
- `processed/`: Cleaned and preprocessed data
- `interim/`: Intermediate data files generated during processing
- `external/`: External data sources and reference data

# 1.1 Data Organization
## 1.1.1 Raw Data
- Original dataset files
- Data in its initial format
- Unmodified source data

## 1.1.2 Processed Data
- Cleaned and normalized data
- Feature-engineered datasets
- Ready-to-use data for modeling

## 1.1.3 Interim Data
- Temporary data files
- Intermediate processing results
- Data checkpoints

# 1.2 Data Management
- All data files should be tracked using Git LFS
- Raw data should never be modified
- Processed data should be reproducible from raw data
- Data versioning is maintained through file naming conventions 