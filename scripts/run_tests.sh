#!/bin/bash

#for seed in {37..100}; do
#  echo "test seed ${seed}"
#  python src/train.py experiment=brain_planes tags='["benchmark"]' \
#    seed="${seed}" trainer.deterministic=True
#done

#for seed in {70..99}; do
#  echo "test seed ${seed}"
#  python src/train.py experiment=video_quality tags='["test"]' \
#    data.train_val_split="0.1" data.train_val_split_seed="${seed}" \
#    +logger.wandb.notes="t ${seed}"
#done

#declare -a arr=(65)
#
#for seed in "${arr[@]}"; do
#  for i in {10..14}; do
#    echo "test seed ${i}"
#    python src/train.py experiment=video_quality tags='["test"]' \
#      data.train_val_split="0.1" data.train_val_split_seed="${seed}" \
#      +logger.wandb.notes="t ${seed} ${i}"
#  done
#done

# densenet169
# mobilenet_v3_small mobilenet_v3_large
# efficientnet_v2_s efficientnet_v2_m
# resnet18 resnet34 resnet50 resnet101 resnet152
# resnext50_32x4d resnext101_32x8d resnext101_64x4d

#python src/train.py experiment=brain_planes tags='["nets"]' \
#  model.net_spec.name="densenet169"


python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
  hydra.sweeper.ckpt="best"
python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
  hydra.sweeper.ckpt="best"
python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
  hydra.sweeper.ckpt="best"

python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
  hydra.sweeper.ckpt="last"
python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
  hydra.sweeper.ckpt="last"
python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
  hydra.sweeper.ckpt="last"


#declare -a arr=("mobilenet_v3_small" "mobilenet_v3_large" "efficientnet_v2_s" "efficientnet_v2_m" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "resnext50_32x4d" "resnext101_32x8d" "resnext101_64x4d")
#
#for model in "${arr[@]}"; do
#  for i in {6..15}; do
#    echo "test model ${model} ${i}"
#    python src/train.py experiment=brain_planes tags='["nets"]' \
#      model.net_spec.name="${model}" \
#      +logger.wandb.notes="t ${i}"
#  done
#done

#python src/train.py experiment=video_quality tags='["test"]'

#python src/train.py experiment=video_quality tags='["test"]' \
#  +logger.wandb.notes="delete it" \
#  trainer.min_epochs=0 trainer.max_epochs=1

#python src/train.py experiment=brain_planes tags='["test"]' \
#  model.net_spec.name="mobilenet_v3_small" \
#  +logger.wandb.notes="delete it" \
#  trainer.min_epochs=0 trainer.max_epochs=1

#python src/train.py experiment=brain_planes tags='["test"]' \
#  model.masks="[[0, 0, 0, 0, 0]]" \
#  +logger.wandb.notes="1 models" \
#  model.net_spec.name="efficientnet_v2_s"

#bash scripts/semi-supervised-learning.sh 10 "0.95"

#python scripts/label_videos.py video_dataset_dir="US_VIDEOS_0.3_ssl" min_prob=0.3 prob_norm=1.0 model_path="logs/train/runs/2023-02-15_10-43-13"

#python src/train.py experiment=brain_planes tags='["test"]' \
#  trainer.max_epochs=3 \
#  +trainer.limit_train_batches=0.01 +trainer.limit_val_batches=0.05 +trainer.limit_test_batches=0.05

#python src/train.py experiment=brain_planes tags='["forest"]' \
#  data.batch_size=16 +trainer.accumulate_grad_batches=2
#
#python src/train.py experiment=brain_planes tags='["forest"]' \
#  data.batch_size=16 +trainer.accumulate_grad_batches=2 \
#  data.video_dataset=true data.video_dataset_dir='US_VIDEOS_0.3'
#
#python src/train.py experiment=brain_planes tags='["forest"]' \
#  data.batch_size=16 +trainer.accumulate_grad_batches=2 \
#  data.video_dataset=true data.video_dataset_dir='US_VIDEOS_0.3' \
#  callbacks.stochastic_weight_averaging.swa_lrs=1e-04 callbacks.stochastic_weight_averaging.swa_epoch_start=8

#python src/train.py experiment=brain_planes tags='["ssl"]' \
#  +logger.wandb.name="ssl-it-10.0" \
#  data.video_dataset=true data.video_dataset_dir="US_VIDEOS_0.4_ssl" \
#  trainer.min_epochs=0 trainer.max_epochs=1
#
#python scripts/label_videos.py video_dataset_dir="US_VIDEOS_0.4_ssl" min_prob="0.95"
