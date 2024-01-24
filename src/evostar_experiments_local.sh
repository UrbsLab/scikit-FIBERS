#!/bin/bash
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold False --evolving-probability 0 --threshold 0 > run_history/exp1.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0 --threshold 0 > run_history/exp2.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0.5 --threshold 0 > run_history/exp3.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 1 --threshold 0 > run_history/exp4.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold False --evolving-probability 0 --threshold 1 > run_history/exp5.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0 --threshold 1 > run_history/exp6.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0.5 --threshold 1 > run_history/exp7.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 1 --threshold 1 > run_history/exp8.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold False --evolving-probability 0 --threshold 2 > run_history/exp9.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0 --threshold 2 > run_history/exp10.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0.5 --threshold 2 > run_history/exp11.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 1 --threshold 2 > run_history/exp12.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold False --evolving-probability 0 --threshold 3 > run_history/exp13.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0 --threshold 3 > run_history/exp14.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 0.5 --threshold 3 > run_history/exp15.log  & \
python evostar_experiment.py --save-folder run_history/ --mutation-strategy Simple --adaptable-threshold True --evolving-probability 1 --threshold 3 > run_history/exp16.log  & \