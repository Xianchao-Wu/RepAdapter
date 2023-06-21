#!/bin/sh
echo "start"
devices=1 #$1
method='repattn'
dim=8
scale=0
model='vit_base_patch16_224_in21k'
bash scripts/exp.sh $devices $method  $dim  $scale $model
#2>&1 | tee ./logs/repadapter-$method.log
