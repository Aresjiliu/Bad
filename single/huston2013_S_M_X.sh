#!/usr/bin/env bash

python huston2013_single_train.py '../data/Huston2013' 'ms' 1 0 > single_ms.out 2>&1 &
python huston2013_single_train.py '../data/Huston2013' 'hsi' 1 0 > single_hsi.out 2>&1 &
python huston2013_single_train.py '../data/Huston2013' 'lidar' 1 0 > single_lidar.out 2>&1 &