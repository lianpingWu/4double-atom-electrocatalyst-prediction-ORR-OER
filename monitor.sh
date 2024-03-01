#!/bin/bash

sample_path="./dataset/final/"

clear
for flag in "BE" "OER" "ORR"
    do
        python -u ./TopoCata/monitor/monitor.py $sample_path $flag $load_epoch
    done

exit 0