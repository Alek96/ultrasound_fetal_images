#!/bin/bash

#python src/train.py -m hparams_search=dende_net_optuna experiment=dense_net model.net.densenet_name=densenet169 data.train_val_split_seed=9208 tags='["dense_net","optuna"]'
#python src/train.py experiment=dense_net tags='["dense_net","_240_"]'

python src/train.py experiment=brain_planes tags='["seed-0.1"]' data.train_val_split="0.1" data.train_val_split_seed="9959" +logger.wandb.notes="test 7"
python src/train.py experiment=brain_planes tags='["seed-0.1"]' data.train_val_split="0.1" data.train_val_split_seed="9959" +logger.wandb.notes="test 8"
python src/train.py experiment=brain_planes tags='["seed-0.1"]' data.train_val_split="0.1" data.train_val_split_seed="9959" +logger.wandb.notes="test 9"

declare -a arr=(3084 9456 2696 2086 4063 9126)

for seed in "${arr[@]}"; do
  for i in {4..9}; do
    echo "test seed ${i}"
    python src/train.py experiment=brain_planes tags='["seed-0.1"]' \
      data.train_val_split="0.1" data.train_val_split_seed="${seed}" \
      +logger.wandb.notes="test ${i}"
  done
done

declare -a arr=(943 9787 4935 6588 6893 697 6347 5785 4 7765)

for seed in "${arr[@]}"; do
  for i in {1..9}; do
    echo "test seed ${i}"
    python src/train.py experiment=brain_planes tags='["seed-0.1", "seed-0.15"]' \
      data.train_val_split="0.15" data.train_val_split_seed="${seed}" \
      +logger.wandb.notes="test ${i}"
  done
done

#python src/train.py experiment=brain_planes tags='["test"]' \
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