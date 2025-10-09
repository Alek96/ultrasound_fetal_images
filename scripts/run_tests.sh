#!/bin/bash

#python src/head_segmentation_train.py experiment=head_segmentation \
#  debug=fdr

#python src/head_segmentation_train.py experiment=head_segmentation \
#  tags='["rerun"]' \
#  +logger.wandb.notes="rerun" \
#  logger.wandb.group="MobileNetV4" \
#  model.optimizer.weight_decay="1e-5" \
#  model.optimizer.amsgrad="True" \
#  seed=1838639280 \
#  clean-up=false

for i in {10..200}; do
  echo "test ${i}"
  python src/head_segmentation_train.py experiment=head_segmentation \
    tags='["test10"]' \
    +logger.wandb.notes="t ${i}" \
    logger.wandb.group="MobileNetV4"
done

#weights=("1e-5")
#num_weights=${#weights[@]}
#amsgrads=("True")
#num_amsgrads=${#amsgrads[@]}
#tests=100
#
#for (( w=0; w<num_weights; w++ )); do
#  for (( a=0; a<num_amsgrads; a++ )); do
#    for (( t=20; t<tests; t++ )); do
#      python src/head_segmentation_train.py experiment=head_segmentation \
#        tags='["test7"]' \
#        +logger.wandb.notes="t $t" \
#        logger.wandb.group="MobileNetV4" \
#        model.optimizer.weight_decay="${weights[$w]}" \
#        model.optimizer.amsgrad="${amsgrads[$a]}"
#    done
#  done
#done

#models=("tu-tf_efficientnetv2_s.in21k")
#num_models=${#models[@]}
#tests=20
#
#for (( m=0; m<num_models; m++ )); do
#  for (( t=0; t<tests; t++ )); do
#    python src/head_segmentation_train.py experiment=head_segmentation \
#      tags='["test7"]' \
#      +logger.wandb.notes="t $t" \
#      logger.wandb.group="EfficientNetV2" \
#      model.model.encoder_name="${models[$m]}" \
#      +trainer.accumulate_grad_batches=4 \
#      data.batch_size=16
#  done
#done


#python src/brain_planes_train.py experiment=brain_planes \
#  debug=fdr

#python src/brain_planes_train.py experiment=brain_planes \
#  tags='["test2"]' \
#  +logger.wandb.notes="rerun" \
#  seed=820096712 \
#  clean-up=false

#for i in {1..200}; do
#  echo "test ${i}"
#  python src/brain_planes_train.py experiment=brain_planes \
#    tags='["test5"]' \
#    +logger.wandb.notes="t ${i}"
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


#for i in {1..20}; do
#  echo "test ${i}"
#  python src/video_quality_train.py experiment=video_quality tags='["test20"]' \
#    +logger.wandb.notes="t 22 ${i}" \
#    data.dataset_name="US_VIDEOS_tran_0500" \
#    logger.wandb.group="g 22"
#done
