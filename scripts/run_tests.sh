#!/bin/bash

#python src/head_segmentation_train.py experiment=head_segmentation \
#  debug=fdr

#python src/head_segmentation_train.py experiment=head_segmentation \
#  tags='["test10", "rerun"]' \
#  +logger.wandb.notes="rerun" \
#  logger.wandb.group="MobileNetV4" \
#  seed=2314826881 \
#  cleanup_mode=none

#for i in {1..100}; do
#  echo "test ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation \
#    tags='["test-05"]' \
#    +logger.wandb.notes="t ${i}"
#done


#weights=("1e-2" "1e-3" "1e-4" "1e-5")
#num_weights=${#weights[@]}
#amsgrads=("False" "True")
#num_amsgrads=${#amsgrads[@]}
#tests=10
#
#for (( w=0; w<num_weights; w++ )); do
#  for (( a=0; a<num_amsgrads; a++ )); do
#    for (( t=1; t<=tests; t++ )); do
#      echo "test ${t}, weights ${weights[$w]}, amsgrad ${amsgrads[$a]}"
#      python src/head_segmentation_train.py experiment=head_segmentation \
#        tags='["test-03"]' \
#        +logger.wandb.notes="t $t" \
#        logger.wandb.group="${weights[$w]} ${amsgrads[$a]}" \
#        model.optimizer.weight_decay="${weights[$w]}" \
#        model.optimizer.amsgrad="${amsgrads[$a]}"
#    done
#  done
#done




# dice_ce       -> smooth=1e-6, 1
# focal         -> alpha: 0.25 0.5 0.75
# dice_focal    -> alpha: 0.25 0.5 0.75
# tversky       -> alpha: 0.6, beta: 0.4 -> alpha: 0.7, beta: 0.3
# focal_tversky -> alpha: 0.6, beta: 0.4 -> alpha: 0.7, beta: 0.3


models=("manet" "fpn")
num_models=${#models[@]}

dice_smooth=("1e-6" "1.0")
num_dice_smooth=${#dice_smooth[@]}

focal_alpha=("0.25" "0.50" "0.75")
num_focal_alpha=${#focal_alpha[@]}

tversky_alpha=("0.5" "0.6" "0.7" "0.4" "0.3")
tversky_beta=("0.5" "0.4" "0.3" "0.6" "0.7")
num_tversky_alpha=${#tversky_alpha[@]}

tests=10



for (( t=1; t<=tests; t++ )); do
  for (( m=0; m<num_models; m++ )); do
    # ${models[$m]} is the model name

    for (( s=0; s<num_dice_smooth; s++ )); do
      # ${dice_smooth[$s]}

      echo "test ${models[$m]} dice_ce ${dice_smooth[$s]} ${t}"
      python src/head_segmentation_train.py experiment="head_segmentation_${models[$m]}" \
        +model/criterion@model.criterion=dice_ce \
        model.criterion.smooth="${dice_smooth[$s]}" \
        tags='["test-21"]' \
        +logger.wandb.notes="t ${t}" \
        logger.wandb.group="${models[$m]} dice_ce ${dice_smooth[$s]}"
    done

    for (( a=0; a<num_focal_alpha; a++ )); do
      # ${focal_alpha[$a]}

      echo "test ${models[$m]} focal ${focal_alpha[$a]} ${t}"
      python src/head_segmentation_train.py experiment="head_segmentation_${models[$m]}" \
        +model/criterion@model.criterion=focal \
        model.criterion.alpha="${focal_alpha[$a]}" \
        tags='["test-21"]' \
        +logger.wandb.notes="t ${t}" \
        logger.wandb.group="${models[$m]} focal ${focal_alpha[$a]}"
    done

    for (( a=0; a<num_focal_alpha; a++ )); do
      # ${focal_alpha[$a]}

      echo "test ${models[$m]} dice_focal ${focal_alpha[$a]} ${t}"
      python src/head_segmentation_train.py experiment="head_segmentation_${models[$m]}" \
        +model/criterion@model.criterion=dice_focal \
        model.criterion.alpha="${focal_alpha[$a]}" \
        tags='["test-21"]' \
        +logger.wandb.notes="t ${t}" \
        logger.wandb.group="${models[$m]} dice_focal ${focal_alpha[$a]}"
    done

    for (( a=0; a<num_tversky_alpha; a++ )); do
      # ${tversky_alpha[$a]}
      # ${tversky_beta[$a]}

      echo "test ${models[$m]} tversky ${tversky_alpha[$a]}-${tversky_beta[$a]} ${t}"
      python src/head_segmentation_train.py experiment="head_segmentation_${models[$m]}" \
        +model/criterion@model.criterion=tversky \
        model.criterion.alpha="${tversky_alpha[$a]}" \
        model.criterion.beta="${tversky_beta[$a]}" \
        tags='["test-21"]' \
        +logger.wandb.notes="t ${t}" \
        logger.wandb.group="${models[$m]} tversky ${tversky_alpha[$a]}-${tversky_beta[$a]}"
    done

    for (( a=0; a<num_tversky_alpha; a++ )); do
      # ${tversky_alpha[$a]}
      # ${tversky_beta[$a]}

      echo "test ${models[$m]} focal_tversky ${tversky_alpha[$a]}-${tversky_beta[$a]} ${t}"
      python src/head_segmentation_train.py experiment="head_segmentation_${models[$m]}" \
        +model/criterion@model.criterion=focal_tversky \
        model.criterion.alpha="${tversky_alpha[$a]}" \
        model.criterion.beta="${tversky_beta[$a]}" \
        tags='["test-21"]' \
        +logger.wandb.notes="t ${t}" \
        logger.wandb.group="${models[$m]} focal_tversky ${tversky_alpha[$a]}-${tversky_beta[$a]}"
    done

  done
done

#for i in {1..1}; do
#  echo "test FPN ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_fpn \
#    tags='["test-21"]' \
#    +logger.wandb.notes="t ${i}" \
#    debug=fdr
#done

# Models

#for i in {1..5}; do
#  echo "test U-Net++ ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_unetpp \
#    tags='["test-05"]' \
#    +logger.wandb.notes="t ${i}"
#done

#for i in {1..8}; do
#  echo "test Attention U-Net ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_attention_unet \
#    tags='["test-20"]' \
#    +logger.wandb.notes="t ${i}"
#done
#
#for i in {1..8}; do
#  echo "test MAnet ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_manet \
#    tags='["test-20"]' \
#    +logger.wandb.notes="t ${i}"
#done
#
#for i in {1..8}; do
#  echo "test FPN ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_fpn \
#    tags='["test-20"]' \
#    +logger.wandb.notes="t ${i}"
#done
#
#for i in {1..8}; do
#  echo "test DeepLabV3+ ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_deeplabv3plus \
#    tags='["test-20"]' \
#    +logger.wandb.notes="t ${i}"
#done

# Backbones

#for i in {1..5}; do
#  echo "test EfficientNet-V2-S ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_efficientnetv2s \
#    tags='["test-10"]' \
#    +logger.wandb.notes="t ${i}"
#done

#for i in {1..5}; do
#  echo "test ConvNeXt-Tiny ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_convnext_tiny \
#    tags='["test-11"]' \
#    +logger.wandb.notes="t ${i}"
#done

#for i in {1..5}; do
#  echo "test ConvNeXt-V2-Tiny ${i}"
#  python src/head_segmentation_train.py experiment=head_segmentation_convnextv2_tiny \
#    tags='["test-12"]' \
#    +logger.wandb.notes="t ${i}"
#done

#weights=("1e-2" "1e-3" "1e-4" "1e-5")
#num_weights=${#weights[@]}
#amsgrads=("False" "True")
#num_amsgrads=${#amsgrads[@]}
#tests=10
#
#for (( w=0; w<num_weights; w++ )); do
#  for (( a=0; a<num_amsgrads; a++ )); do
#    for (( t=1; t<=tests; t++ )); do
#      echo "test ${t}, weights ${weights[$w]}, amsgrad ${amsgrads[$a]}"
#      python src/head_segmentation_train.py experiment=head_segmentation \
#        tags='["test-03"]' \
#        +logger.wandb.notes="t $t" \
#        logger.wandb.group="${weights[$w]} ${amsgrads[$a]}" \
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
#  cleanup_mode=none

#for i in {1..200}; do
#  echo "test ${i}"
#  python src/brain_planes_train.py experiment=brain_planes \
#    tags='["test7"]' \
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
