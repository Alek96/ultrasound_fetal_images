#!/bin/bash

#for i in {1..200}; do
#  echo "test 1 ${i}"
#  python src/brain_planes_train.py experiment=_ra_vf_15_01_12 \
#    tags='["tta-softmax-test", "rerun"]' \
#    +logger.wandb.notes="t ${i}" \
#    logger.wandb.group="_ra_vf_20_01_12"
#done

#python src/brain_planes_train.py experiment=brain_planes_82 \
#    tags='["tta-softmax-test"]' \
#    +logger.wandb.notes="t 15 9 rerun" \
#    logger.wandb.group="_ra_vf_20_01_12"

#python src/create_quality_dataset.py experiment=create_quality_dataset_0500 \
#  model_path="logs/train/runs/2024-06-22_21-48-38"  # glowing-sea-2849
#
#python src/video_quality_train.py experiment=video_quality tags='["test12"]' \
#    +logger.wandb.notes="t 12 58 2" \
#    data.dataset_name="US_VIDEOS_tran_0500" \
#    logger.wandb.group="g 500 12" \
#    seed=2713491474

#for i in {1..100}; do
#  echo "test ${i}"
#  python src/video_quality_train.py experiment=video_quality tags='["test15"]' \
#    +logger.wandb.notes="t 15 ${i}" \
#    data.dataset_name="US_VIDEOS_tran_0500" \
#    logger.wandb.group="g 500 15" \
#    data.normalize=false \
#    model.criterion._target_=torch.nn.MSELoss
#done

#for i in {1..100}; do
#  echo "test ${i}"
#  python src/video_quality_train.py experiment=video_quality tags='["test18"]' \
#    +logger.wandb.notes="t 18 ${i}" \
#    data.dataset_name="US_VIDEOS_tran_0500" \
#    logger.wandb.group="g 500 18" \
#    data.normalize=true \
#    model.criterion._target_=torch.nn.MSELoss
#done


for i in {1..20}; do
  echo "test ${i}"
  python src/video_quality_train.py experiment=video_quality tags='["test20"]' \
    +logger.wandb.notes="t 22 ${i}" \
    data.dataset_name="US_VIDEOS_tran_0500" \
    logger.wandb.group="g 22"
done
