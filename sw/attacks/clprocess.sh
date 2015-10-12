#!/bin/bash

for fname in clLev-defdata2-2-
do
	for run in {0..100}
	do
		cp experiment08/$fname$run clLev_$run.lev
	done
	rm clLev_acc
	rm cl_matrix
	rm cltor_matrix
	./clcantor2matrix clmatrix clLev_*.lev
	./clgen_gamma_matrix 1 clmatrix cltor_matrix
	echo "Starting training:"
	ruby svmdotest.rb
	echo $fname >> clprocess.results
	grep Accuracy clLev_acc >> clprocess.results
done
