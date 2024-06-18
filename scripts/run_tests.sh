#!/bin/bash


python src/brain_planes_train.py experiment=brain_planes_82 \
    tags='["tta-softmax-test"]' \
    +logger.wandb.notes="t 15 9 rerun" \
    logger.wandb.group="_ra_vf_20_01_12"

python src/create_quality_dataset.py experiment=create_quality_dataset_0500 \
  model_path="logs/train/multiruns/2023-11-07_21-10-40/0" # frosty-forest-2691 0.8066

for i in {1..200}; do
  echo "test 1 ${i}"
  python src/video_quality_train.py experiment=video_quality tags='["test9"]' \
    +logger.wandb.notes="t 9 2 ${i}" \
    data.dataset_name="US_VIDEOS_tran_0500" \
    logger.wandb.group="g 500 9"
done
