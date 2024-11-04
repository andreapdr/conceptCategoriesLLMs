#!bin/bash

python compute_metrics_unibo.py --datapath italian/formatted/temp_00/idefics2-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/idefics2-it-visual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/llama3.1-70b-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/llama3.1-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/llama3.2-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/llava-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/llava-it-visual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/mistral-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/mixtral-it-textual-temp_00_alliters.csv
python compute_metrics_unibo.py --datapath italian/formatted/temp_00/nemo-it-textual-temp_00_alliters.csv