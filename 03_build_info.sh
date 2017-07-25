#!/bin/bash
echo "Build info..."

taken="all" #one/all
bound=40 #8~40

for type in "training" "testing"; do
	python build_info.py $type $taken $bound
	echo "Finish building" $type "info!!" 
done
