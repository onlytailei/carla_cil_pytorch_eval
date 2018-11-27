#!/bin/bash

# weathers 1 clear  3 wet  6 hardrain  8 sunset

python run_CIL.py \
  --log-name local_test \
  --weathers 1 \
  --model-path "model_policy/new_dataset_best.pth" \
  --vrg-transfer \
  --vrg-model-transfer "model_transfer/transfer.pth" \
  --visualize
