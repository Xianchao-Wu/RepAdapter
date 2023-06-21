#!/bin/sh
echo "start"
devices=1
#"cuda:0" #$1
method='repblock'
dim=8
scale=0
modelvit="vit_base_patch16_224_in21k"
modelswin="swin_base_patch4_window7_224_in22k"
modelconvnext="convnext_base_22k_224"

#bash scripts/exp.sh $devices $method  $dim  $scale  2>&1 | tee ./logs/repadapter-$method.log

bash scripts/exp.sh $devices $method  $dim  $scale $modelvit

#bash scripts/exp.sh $devices $method  $dim  $scale $modelswin

#bash scripts/exp.sh $devices $method  $dim  $scale $modelconvnext

#2>&1 | tee ./logs/repadapter-$method.log
