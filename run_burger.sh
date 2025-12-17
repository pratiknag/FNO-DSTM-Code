#!/bin/bash

## running simulations for Burgers' equation ##
python src/python_scripts/burger/generate_burger-sample.py
python src/python_scripts/burger/FNO-nonlocal-Burger.py
python src/python_scripts/burger/ConvLSTM-Burger.py
python src/python_scripts/burger/STDK-nonlocal-Burger.py
Rscript src/r_scripts/make_plots_burger.R
