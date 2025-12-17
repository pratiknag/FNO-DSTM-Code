#!/bin/bash

## running comparative models on sst data
python src/python_scripts/Data_application/FNO_sst.py --data="sst" --train
python src/python_scripts/Data_application/ConvLSTM-sst.py --data="sst" --train
python src/python_scripts/Data_application/CDNN-sst.py --data="sst" --train
python src/python_scripts/Data_application/STDK-sst.py --data="sst" --train

Rscript src/r_scripts/make_plots_data_application.R "sst" 47
