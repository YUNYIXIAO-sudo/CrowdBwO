#!/bin/bash

cd /ExpSytheticWorkers
#To the experiment directory

python3 simulation_LP.py
python3 TRbatchedBandit.py
python3 BwK-RRS.py
python3 OSW.py
python3 OSWOP.py
python3 OWOP.py
python3 WSW.py
python3 AW.py