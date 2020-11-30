#!/bin/bash
best=0.13
m=0
rm -Rf models/
rm -f all_result best_result

for i in `seq 10 30`;
do
	for j in `seq 10 30`;
	do
		python3 train.py resources/data.csv -p50 -hl "$i, $j" > /dev/null
		arg_1=`python3 predict.py resources/data.csv models/model_$m.p`
		m=$((m+1))


		python3 train.py data_training.csv -p50 -hl "$i, $j" > /dev/null
		arg_2=`python3 predict.py data_test.csv models/model_$m.p`
		m=$((m+1))


		echo "$i, $j\tResources : $arg_1\tEvaluation: $arg_2"


		st=`echo "$arg_2 < $best" | bc`
		echo "$i, $j\tResources : $arg_1\tEvaluation: $arg_2" >> all_result
		if [ $st -eq 1 ]
		then
			echo "$i, $j\tResources : $arg_1\tEvaluation: $arg_2" >> best_result
		fi
	done
done