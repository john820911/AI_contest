#!/bin/bash
echo "Build dic..."

embed_size=300 #100/300
vocab_size=4000

python build_dic.py $embed_size $vocab_size

echo "Finish building dic!!"
