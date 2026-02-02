#!/bin/bash
for ((i=1; i<10; i++))
do
  python train_tricks.py $i
done
