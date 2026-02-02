#!/bin/bash
for ((i=1; i<2; i++))
do
  python train_tricks.py $i
done
