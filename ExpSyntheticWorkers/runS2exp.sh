#!/bin/bash

cd /ExpSytheticWorkers
#To the experiment directory

python3 simulation_MO.py
python3 TRbatchedBandit-S2.py
python3 BwK-GD.py
python3 OSW-S2.py
python3 OSWOP-S2.py
python3 OWOP-S2.py
python3 WSW-S2.py
python3 AW-S2.py