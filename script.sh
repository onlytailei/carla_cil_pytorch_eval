#!/bin/bash

# weathers 1 clear  3 wet  6 hardrain  8 sunset

python run_CIL.py \
  --log-name local_test \
  --weathers 6 \
  --model-path "model_policy/three_policy_138_best.pth" \
