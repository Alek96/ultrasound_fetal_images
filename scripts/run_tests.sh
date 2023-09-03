#!/bin/bash

#for seed in {37..100}; do
#  echo "test seed ${seed}"
#  python src/train.py experiment=brain_planes tags='["benchmark"]' \
#    seed="${seed}" trainer.deterministic=True
#done

#for seed in {10..99}; do
#  for i in {1..3}; do
#    echo "test seed ${seed} ${i}"
#    python src/train.py experiment=video_quality tags='["test2"]' \
#      data.train_val_split="0.1" data.train_val_split_seed="${seed}" \
#      +logger.wandb.notes="t ${seed} ${i}"
#  done
#  for i in {4..6}; do
#    echo "test seed ${seed} ${i}"
#    python src/train.py experiment=video_quality tags='["test2"]' \
#      data.train_val_split="0.1" data.train_val_split_seed="${seed}" \
#      +logger.wandb.notes="t ${seed} ${i}" \
#      data.transform=true
#  done
#done

#for seed in {1..99}; do
#  echo "test seed ${seed}"
#  python src/train.py experiment=video_quality tags='["test"]' \
#    data.train_val_split="0.1" data.train_val_split_seed="${seed}" \
#    +logger.wandb.notes="t ${seed}"
#done


for i in {41..200}; do
  echo "test 0 ${i}"
  python src/train.py experiment=video_quality tags='["test3"]' \
    +logger.wandb.notes="t 0 ${i}" \
    data.dataset_name="US_VIDEOS_tran_0250_playful-haze-2111" \
    logger.wandb.group="g 250"
done

for i in {1..200}; do
  echo "test 1 ${i}"
  python src/train.py experiment=video_quality tags='["test3"]' \
    +logger.wandb.notes="t 1 ${i}" \
    data.dataset_name="US_VIDEOS_tran_0350_playful-haze-2111" \
    logger.wandb.group="g 350"
done

for i in {41..200}; do
  echo "test 2 ${i}"
  python src/train.py experiment=video_quality tags='["test3"]' \
    +logger.wandb.notes="t 2 ${i}" \
    data.dataset_name="US_VIDEOS_tran_0490_playful-haze-2111" \
    logger.wandb.group="g 490"
done

for i in {41..200}; do
  echo "test 3 ${i}"
  python src/train.py experiment=video_quality tags='["test3"]' \
    +logger.wandb.notes="t 3 ${i}" \
    data.dataset_name="US_VIDEOS_tran_0882_playful-haze-2111" \
    logger.wandb.group="g 882"
done

for i in {41..200}; do
  echo "test 4 ${i}"
  python src/train.py experiment=video_quality tags='["test3"]' \
    +logger.wandb.notes="t 4 ${i}" \
    data.dataset_name="US_VIDEOS_tran_1134_playful-haze-2111" \
    logger.wandb.group="g 1134"
done

rm -rf data/US_VIDEOS_tran_882_playful-haze-2111/data
rm -rf data/US_VIDEOS_tran_1134_playful-haze-2111/data

python src/create_quality_dataset.py

for i in {1..200}; do
  echo "test 5 ${i}"
  python src/train.py experiment=video_quality tags='["test3"]' \
    +logger.wandb.notes="t 5 ${i}" \
    data.dataset_name="US_VIDEOS_tran_2106_playful-haze-2111" \
    logger.wandb.group="g 2106"
done

#python src/create_quality_dataset.py

#python src/train.py experiment=brain_planes_mixup tags='["smooth"]' +logger.wandb.notes="mix-up"
#python src/train.py experiment=brain_planes_mixup tags='["smooth"]' +logger.wandb.notes="mix-up"
#python src/train.py experiment=brain_planes_mixup tags='["smooth"]' +logger.wandb.notes="mix-up"
#
#
#for i in {1..5}; do
#  echo "test ${i}"
#  python src/train.py experiment=brain_planes_mixup tags='["smooth"]' +logger.wandb.notes="mix-up"
#  python src/train.py experiment=brain_planes_smooth tags='["smooth"]' +logger.wandb.notes="smooth" model.criterion.label_smoothing=0.02
#  python src/train.py experiment=brain_planes_smooth tags='["smooth"]' +logger.wandb.notes="clear"
#done

#python src/train.py experiment=brain_planes_smooth tags='["smooth"]'


# densenet169
# mobilenet_v3_small mobilenet_v3_large
# efficientnet_v2_s efficientnet_v2_m
# resnet18 resnet34 resnet50 resnet101 resnet152
# resnext50_32x4d resnext101_32x8d resnext101_64x4d

#python src/train.py experiment=brain_planes tags='["nets"]' \
#  model.net_spec.name="densenet169"


#python src/train.py 'experiment=brain_planes' tags='["benchmark_v2","models"]'


#python src/train.py -m 'experiment=brain_planes' tags='["benchmark_v2","models"]' \
#   model.net_spec.name=densenet169,mobilenet_v3_small,mobilenet_v3_large,efficientnet_v2_s,efficientnet_v2_m,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d \
#   model.optimizer.lr=5e-04
#python src/train.py -m 'experiment=brain_planes' tags='["benchmark_v2","models"]' \
#   model.net_spec.name=densenet169,mobilenet_v3_small,mobilenet_v3_large,efficientnet_v2_s,efficientnet_v2_m,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d \
#   model.optimizer.lr=5e-04
#python src/train.py -m 'experiment=brain_planes' tags='["benchmark_v2","models"]' \
#   model.net_spec.name=densenet169,mobilenet_v3_small,mobilenet_v3_large,efficientnet_v2_s,efficientnet_v2_m,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d \
#   model.optimizer.lr=5e-04
#python src/train.py -m 'experiment=brain_planes' tags='["benchmark_v2","models"]' \
#   model.net_spec.name=densenet169,mobilenet_v3_small,mobilenet_v3_large,efficientnet_v2_s,efficientnet_v2_m,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d \
#   model.optimizer.lr=5e-04
#python src/train.py -m 'experiment=brain_planes' tags='["benchmark_v2","models"]' \
#   model.net_spec.name=densenet169,mobilenet_v3_small,mobilenet_v3_large,efficientnet_v2_s,efficientnet_v2_m,resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d \
#   model.optimizer.lr=5e-04

#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'
#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'
#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'
#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'
#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'
#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'
#python src/train.py -m 'experiment=glob(_*)' tags='["benchmark_v2","tta"]'


#python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
#  hydra.sweeper.ckpt="best"
#python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
#  hydra.sweeper.ckpt="best"
#python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
#  hydra.sweeper.ckpt="best"
#
#python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
#  hydra.sweeper.ckpt="last"
#python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
#  hydra.sweeper.ckpt="last"
#python src/train.py -m progressive_training=brain_planes experiment=brain_planes tags='["prog"]' \
#  hydra.sweeper.ckpt="last"


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
