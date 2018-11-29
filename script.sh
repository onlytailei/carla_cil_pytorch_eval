#!/bin/bash

# weathers 1 clear  3 wet  6 hardrain  8 sunset

python run_CIL.py \
  --log-name local_test \
  --weathers 3 \
  --model-path "model_policy/wet_policy_best.pth" \
  #--vrg-transfer \
  #--b2a \
  #--vrg-model-path "model_transfer/clear2rain_cg.pth" \
  #--visualize
