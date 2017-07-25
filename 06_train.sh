#!/bin/bash
echo "Train model..."

version="attention1"
python $version/train.py

echo "Finish training model!!"
