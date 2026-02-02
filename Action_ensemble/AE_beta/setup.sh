#!/bin/bash
for ((i=1; i<8; i++))
do
  python train_tricks.py $i
done
