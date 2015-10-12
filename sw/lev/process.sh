#!/bin/bash

folder=./newcelltraces

websites=10
trials=10

for set in 3
do
	for metric in 50
	do
	#this part takes a few hundred CPU hours at websites=100 trials=40. send each part to a different core. 
	totalparts=10
		for part in $(seq 1 $((totalparts-1)))
		do
			echo "$((totalparts-1))"
			echo "./Lev $folder$set/ $websites $trials $metric $part $totalparts cantor_Tor_${websites}_${trials}_${set}_${metric}_${part}"
			./Lev $folder$set/ $websites $trials $metric $part $totalparts cantor_Tor_${websites}_${trials}_${set}_${metric}_${part}
		done
		echo "Working on set $set metric $metric:"
		./cantor2matrix matrix $websites $trials ./cantor_Tor_${websites}_${trials}_${set}_${metric}_*
		echo "Starting training:"
		ruby svmdotest.rb $websites $trials
		cp tor_result tor_result$set$metric
		grep Accuracy tor_result$set$metric
		#rm gnorm_cus*
		#rm tor_100_40_gnorm_matrix
		#rm tor_result
	done
done
