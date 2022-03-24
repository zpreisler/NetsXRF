#!/bin/bash


for i in `seq 5`
do
	for j in `seq 3`
	do
		nice -19 ../../train.py config_cuda.yaml -n ResNet1_$i\_$j &
	done
	wait
done
