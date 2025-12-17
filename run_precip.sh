#!/bin/bash

## running comparative models on sst data
python src/python_scripts/Data_application/FNO_sst.py --data="precip" --train
python src/python_scripts/Data_application/ConvLSTM-sst.py --data="precip" --train
python src/python_scripts/Data_application/CDNN-sst.py --data="precip" --train
python src/python_scripts/Data_application/STDK-sst.py --data="precip" --train

Rscript src/r_scripts/make_plots_data_application.R "precip" 10
