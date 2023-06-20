#!/bin/sh
echo "start"
devices="cuda:0" #$1
method='repblock'
dim=8
scale=0

#bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method.log
bash scripts/exp.sh $devices $method  $dim  $scale
#2>&1 | tee ./logs/repadapter-$method.log
