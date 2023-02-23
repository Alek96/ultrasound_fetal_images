#!/bin/bash

ssl_version=${1:-0}
min_prob=${2:-"0.95"}
US_VIDEOS_FOLDER="US_VIDEOS_0.4_ssl"

#python src/train.py experiment=brain_plane_dense_net tags='["ssl"]' \
#  +logger.wandb.name="ssl-it-${ssl_version}.0" \
#  data.video_dataset=true data.video_dataset_dir="${US_VIDEOS_FOLDER}" \
#  extras.plot_probabilities.video_dataset_dir="${US_VIDEOS_FOLDER}" \
#  trainer.min_epochs=0 trainer.max_epochs=1

for i in {6..11}; do
  echo "--------------------------------------------------------------------"
  echo "--------------------------------------------------------------------"
  echo "--------------------------------------------------------------------"
  echo "Iteration ${i}"
  echo "Label video dataset"
  python scripts/label_videos.py video_dataset_dir="${US_VIDEOS_FOLDER}" min_prob="${min_prob}"

  echo "--------------------------------------------------------------------"
  echo "Train on video dataset"
  python src/train.py experiment=brain_plane_dense_net tags='["ssl"]' \
    +logger.wandb.name="ssl-it-${ssl_version}.${i}" \
    data.video_dataset=true data.video_dataset_dir="${US_VIDEOS_FOLDER}" \
    extras.plot_probabilities.video_dataset_dir="${US_VIDEOS_FOLDER}" \
    trainer.min_epochs="${i}" trainer.max_epochs="$((i + 1))" \
    ckpt_path="$(find logs/train/runs/*/checkpoints/last.ckpt | tail -1)" \
    trainer.val_check_interval=0.25
  #    callbacks.early_stopping.patience=40
done
