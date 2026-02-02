#!/bin/bash
for ((i=1; i<7; i++))
do
  python train_tricks.py $i
  # python test.py $i

done
