#!/bin/bash
dimensions=(1 2 5 10 20 50)
ensemble_sizes=(10 20 50 100 200 500 1000 2000)
num_learning_rules=(0 1 2 3 4)

for d in "${dimensions[@]}"
do
for e in "${ensemble_sizes[@]}"
do
for l in "${num_learning_rules[@]}"
do
python profile_communication_channel.py $d $e $l $1
done
done
done