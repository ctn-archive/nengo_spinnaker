#!/bin/bash
dimensions=(1 2 5 10 20 50)
ensemble_sizes=(10 20 50 100 200 500 1000 2000)

for d in "${dimensions[@]}"
do
for e in "${ensemble_sizes[@]}"
do
python profile_communication_channel.py $d $e
done
done