#!/bin/bash

for i in `seq 5`
do
	../train.py config_cuda.yaml -n ResNet1_$i
done
