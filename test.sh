#!/bin/bash
best=0.13
m=0
rm -Rf models/
rm -f all_result best_result

for i in `seq 1 30`;
do
	for j in `seq 1 30`;
	do
		python3 train.py resources/data.csv -e100 -p20 -hl "$i, $j" > /dev/null
		arg1=`python3 predict.py resources/data.csv models/model_$m.p`
		m=$((m+1))

		python3 train.py data_training.csv -e100 -p20 -hl "$i, $j" > /dev/null
		arg=`python3 predict.py data_test.csv models/model_$m.p`
		n=$((m+1))
		echo "$i, $j\tResources : $arg1\tEvaluation: $arg\tmodels/model_$m.p"


		st=`echo "$arg < $best" | bc`
		echo "$i, $j\tResources : $arg1\tEvaluation: $arg\tmodels/model_$m.p" >> all_result
		if [ $st -eq 1 ]
		then
			best=$arg
			echo "$i, $j\tResources : $arg1\tEvaluation: $arg\tmodels/model_$m.p" >> best_result
		fi
	done
done