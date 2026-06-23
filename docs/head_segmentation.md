# Fetal Head Segmentation — Experiment Tracker

> Personal experiment log for fetal head segmentation on ultrasound images
> with a derived binary classification output (Brain / Not A Brain).
>
> **Legend:** ✅ selected  |  🧪 tested  |  📋 to test  |  ❌ rejected

______________________________________________________________________

## Task Overview

Binary segmentation of the fetal head in 2D ultrasound images.
A secondary **classification label** is derived from the predicted mask:
if ≥ 5 % of pixels are positive the image is classified as "Brain", otherwise "Not A Brain".

Input: single-channel (grayscale) ultrasound image, resized to a fixed resolution.
Output: single-channel binary mask (same spatial size as input).

```mermaid
flowchart LR
  image[Ultrasound frame]
  unet[Encoder-decoder network]
  mask[Head probability map]
  frameLabel[Frame brain label]
  image --> unet --> mask --> frameLabel
```

______________________________________________________________________

## 1. Architecture

### 1.1 Segmentation Model

| Technique                         | Status | Description                                                                                                                                                     | Result                                                                                |
| --------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| U-Net                             | ✅     | Encoder–decoder with skip connections. Proven baseline for medical image segmentation — skip connections preserve spatial detail lost during downsampling.      | Selected via segmentation_models_pytorch library. Strong Dice scores out of the box.  |
| U-Net++                           | 📋     | Nested dense skip connections between encoder and decoder. Can improve fine-grained boundary recovery by reusing intermediate features at multiple resolutions. | Queued as `head_segmentation_unetpp`.                                                 |
| DeepLabV3+                        | 📋     | Atrous spatial pyramid pooling (ASPP) captures multi-scale context without losing resolution. Good when the object of interest appears at varying scales.       | Queued as `head_segmentation_deeplabv3plus`.                                          |
| Attention U-Net                   | 📋     | Adds attention gates to standard U-Net skip connections. Suppresses irrelevant encoder features, focusing the decoder on the target region.                     | Queued as `head_segmentation_attention_unet` (Unet + `decoder_attention_type: scse`). |
| FPN (Feature Pyramid Network)     | 📋     | Lightweight top-down decoder that merges multi-scale features. Faster inference than U-Net; worth testing for speed vs. accuracy trade-off.                     | Queued as `head_segmentation_fpn`.                                                    |
| MAnet (Multi-scale Attention Net) | 📋     | Attention-guided multi-scale decoder from SMP. Designed specifically for medical image segmentation tasks.                                                      | Queued as `head_segmentation_manet`.                                                  |

### 1.2 Encoder (Backbone)

| Technique              | Status | Description                                                                                                                        | Result                                                                          |
| ---------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| MobileNetV2            | 🧪     | Lightweight inverted-residual CNN. Low computational cost, ImageNet-pretrained. Good for fast iteration.                           | Default backbone (`mobilenet_v2`). Replaced by MobileNetV4 for better accuracy. |
| MobileNetV4-Conv-Small | ✅     | Next-gen mobile backbone with improved inverted residuals and Universal Inverted Bottleneck. Better accuracy/speed Pareto than V2. | `tu-mobilenetv4_conv_small`, ImageNet weights. Current best encoder.            |
| EfficientNet-V2-S      | 📋     | Compound-scaled CNN with fused MBConv blocks. Faster training and better accuracy than EfficientNet-V1 at similar FLOPS.           | Queued as `head_segmentation_efficientnetv2s` (`tu-tf_efficientnetv2_s`).       |
| ConvNeXt-Tiny          | 📋     | Modernised pure-ConvNet design competitive with vision transformers. Heavier than MobileNets but extracts stronger features.       | Queued as `head_segmentation_convnext_tiny` (`tu-convnext_tiny`).               |
| ConvNeXt-V2-Tiny       | 📋     | ConvNeXt-V2 adds Global Response Normalisation and FCMAE pre-training. Improves over V1 on dense prediction at the same size.      | Queued as `head_segmentation_convnextv2_tiny` (`tu-convnextv2_tiny`).           |

### 1.3 Encoder Pre-training

| Technique           | Status | Description                                                                                                                                       | Result                       |
| ------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| ImageNet supervised | ✅     | Standard supervised pre-training on ImageNet-1k. Provides general low/mid-level features that transfer well to medical images despite domain gap. | `encoder_weights: imagenet`. |

______________________________________________________________________

## 2. Loss Function

| Technique                      | Status | Description                                                                                                                                                                                                                                                        | Result                                                                  |
| ------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| BinaryDice only                | 🧪     | Dice loss alone. Optimises region overlap but can be noisy for very small foreground regions due to lack of per-pixel gradient signal.                                                                                                                             | Available as `BinaryDiceLoss`. Less stable training than combined loss. |
| BinaryDice + BCE (combined)    | ✅     | Sum of Dice loss and BCE-with-logits. Dice handles class imbalance by optimising overlap directly; BCE adds stable per-pixel gradients that prevent training collapse.                                                                                             | `BinaryDiceCrossEntropyLoss`, smooth=1e-6.                              |
| Dice + BCE with pos_weight     | 📋     | Uses the existing `pos_weight` parameter in BCEWithLogitsLoss to upweight foreground pixels in the BCE component. Simple way to address pixel-level class imbalance without changing loss architecture.                                                            | —                                                                       |
| Focal Loss                     | 📋     | Down-weights well-classified (easy) pixels, focusing training on hard examples near boundaries. Useful when background overwhelms foreground in the loss.                                                                                                          | —                                                                       |
| Dice + Focal (combined)        | 📋     | Replaces BCE with Focal in the combined loss. Keeps Dice's overlap objective while adding hard-example mining — potentially better boundary learning.                                                                                                              | —                                                                       |
| Boundary Loss (distance-based) | 📋     | Computes loss using the distance transform of the ground-truth boundary. Directly optimises boundary accuracy — critical for downstream head circumference measurement. Complementary to Dice; typically added as a weighted term after initial Dice-only warm-up. | —                                                                       |
| Tversky Loss                   | 📋     | Generalisation of Dice with separate alpha/beta parameters to weight FP and FN independently. Allows explicit tuning of the precision/recall trade-off at the loss level.                                                                                          | —                                                                       |

______________________________________________________________________

## 3. Optimiser

| Technique | Status | Description                                                                                                                                                         | Result                                                                   |
| --------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Adam      | ✅     | Standard adaptive moment estimation. Amsgrad fixes a convergence issue in vanilla Adam by keeping the max of past squared gradients.                                | lr=1e-3, weight_decay=1e-5, amsgrad=True.                                |
| AdamW     | 🧪     | Decoupled weight decay — applies decay directly to weights rather than through gradient. Theoretically more correct L2 regularisation than Adam's coupled approach. | Adam trained better for MobileNetV4. Revisit on larger backbones/models. |
| RAdam     | 📋     | Rectified Adam with built-in adaptive warm-up. Removes the need for manual LR warm-up scheduling during early training steps.                                       | —                                                                        |

______________________________________________________________________

## 4. Learning Rate Schedule

| Technique                   | Status | Description                                                                                                                                             | Result                                    |
| --------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| ReduceLROnPlateau           | ✅     | Reactive schedule that multiplies LR by a factor when the monitored metric stops improving. Simple and robust — adapts to the actual training dynamics. | factor=0.1, patience=5, monitor=val/loss. |
| OneCycleLR                  | 📋     | Super-convergence recipe: ramps LR up then down in a single cycle. Often converges faster and to better minima in fewer epochs.                         | —                                         |
| CosineAnnealingLR           | 📋     | Smooth cosine decay from initial LR to near-zero. Works well when the total number of epochs is fixed and known in advance.                             | —                                         |
| CosineAnnealingWarmRestarts | 📋     | Periodic cosine restarts that re-raise the LR. Can help escape local minima during longer training runs.                                                | —                                         |

______________________________________________________________________

## 5. Data & Preprocessing

### 5.1 Input Resolution

| Technique | Status | Description                                                                                                           | Result                                                     |
| --------- | ------ | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 55 × 80   | 🧪     | Very small resolution for fast prototyping. Quick iteration but loses fine anatomical structures and boundary detail. | Default config value. Replaced by 192×256 for experiments. |
| 192 × 256 | ✅     | Medium resolution balancing detail preservation and GPU memory/speed. Sufficient to capture head boundary contours.   | Current experiment setting. Good trade-off.                |
| 224 × 320 | 📋     | Higher fidelity closer to common ImageNet input sizes. May improve boundary accuracy at moderate extra cost.          | —                                                          |
| 384 × 512 | 📋     | Near-native resolution. Highest accuracy ceiling but significantly more memory and slower training.                   | —                                                          |

### 5.2 Interpolation

| Technique                               | Status | Description                                                                                                                                         | Result                                                |
| --------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Bilinear (image) + Nearest-Exact (mask) | ✅     | Bilinear smoothly downscales image intensities. Nearest-Exact on masks preserves hard label boundaries without introducing interpolation artefacts. | Applied separately per tensor type via custom Resize. |

### 5.3 Aspect Ratio Handling

| Technique                       | Status | Description                                                                                                                     | Result                                                                       |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| PadToAspectRatio + Resize       | ✅     | Pads the image to the target aspect ratio before resizing. Prevents spatial distortion that could misshape the elliptical head. | Zero-padding for both Image and Mask tensors.                                |
| Direct resize (with distortion) | ❌     | Resize without padding. Introduces anisotropic distortion that deforms the elliptical head shape.                               | Rejected — distortion harms segmentation of geometrically regular targets.   |
| Center crop + Resize            | ❌     | Crops to the target aspect ratio. Risk of cutting off the head region when it's near image edges.                               | Rejected — unacceptable data loss for a task where the target is peripheral. |

### 5.4 Normalisation

| Technique                              | Status | Description                                                                                                                                                                                   | Result                                             |
| -------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Scale to [0, 1]                        | 🧪     | Divide pixel values by 255. Simple, no dataset statistics needed, works with any input.                                                                                                       | `ToDtype(scale=True)` to float32.                  |
| ImageNet stats (mean=0.449, std=0.226) | ✅     | Z-normalisation with ImageNet statistics. Aligns input distribution to what the pretrained encoder's batch-norm running stats expect. High priority given the encoder is ImageNet-pretrained. | Slightly higher avg Dice than [0,1] scaling alone. |
| FetalBrain stats (mean=0.17, std=0.19) | 📋     | Z-normalisation with dataset-specific statistics. Could help convergence by centering the data distribution closer to zero. Test after ImageNet stats if encoder is trained from scratch.     | —                                                  |

______________________________________________________________________

## 6. Data Augmentation

### 6.1 Geometric Augmentations

| Technique            | Status | Description                                                                                                                                  | Result                                     |
| -------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| RandomHorizontalFlip | ✅     | Mirrors the image left-right. Valid because the fetal head has no inherent left/right semantic orientation in ultrasound.                    | p=0.5. Applied jointly to image and mask.  |
| RandomVerticalFlip   | ✅     | Mirrors the image top-bottom. Adds orientation invariance given the variable probe positioning during scanning.                              | p=0.5. Applied jointly to image and mask.  |
| RandomAffine         | ✅     | Combines rotation, translation, and scaling. Simulates natural variation in probe angle, position, and distance to the fetus.                | degrees=±20, translate=10%, scale=1.0–1.2. |
| ElasticTransform     | 📋     | Applies smooth non-rigid deformation. Simulates soft-tissue deformation — the most commonly used augmentation in medical image segmentation. | —                                          |
| RandomPerspective    | 📋     | Mild projective warp. Simulates oblique scanning angles where the probe is tilted relative to the head.                                      | —                                          |

### 6.2 Intensity / Photometric Augmentations

| Technique                      | Status | Description                                                                                                                                  | Result                                                         |
| ------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| RandAugment                    | ✅     | Automated augmentation: randomly samples N transforms at a given magnitude from a search space. Single magnitude knob makes tuning simple.   | magnitude=11.                                                  |
| AutoAugment (ImageNet policy)  | ❌     | Learned augmentation policy from ImageNet. The ImageNet-specific policy may not transfer well to single-channel grayscale ultrasound images. | Not well suited for US domain.                                 |
| TrivialAugmentWide             | ❌     | Applies a single random augmentation per sample (no composition). Simpler than RandAugment, sometimes competitive.                           | RandAugment preferred for its richer augmentation composition. |
| AugMix                         | ❌     | Mixes multiple augmented views with a consistency loss. Improves robustness but adds training overhead.                                      | Overhead not justified for current dataset size.               |
| Random Brightness/Contrast     | 📋     | Explicit control over intensity shifts. Simulates different ultrasound gain and TGC settings across scanners and operators.                  | —                                                              |
| GaussianBlur                   | 📋     | Applies Gaussian smoothing. Simulates defocus and acoustic attenuation artefacts common in ultrasound imaging.                               | —                                                              |
| Speckle Noise (multiplicative) | 📋     | Multiplicative noise model specific to ultrasound. More physically realistic than additive Gaussian noise for US image formation.            | —                                                              |
| CLAHE (as augmentation)        | 📋     | Contrast-Limited Adaptive Histogram Equalisation. Can enhance low-contrast structures; using it as an augmentation adds contrast diversity.  | —                                                              |

### 6.3 Augmentation Strategy

| Technique                    | Status | Description                                                                                                                                  | Result        |
| ---------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| RandAugment (current)        | ✅     | Automated search over augmentation space with a single magnitude parameter. Balances augmentation diversity and simplicity.                  | magnitude=11. |
| CutOut / RandomErasing       | 📋     | Randomly erases rectangular patches. Forces the model to use non-local context rather than relying on a single discriminative region.        | —             |
| MixUp / CutMix               | 📋     | Linearly interpolates or spatially mixes pairs of training samples. Improves calibration and generalisation, especially on smaller datasets. | —             |
| Test-Time Augmentation (TTA) | 📋     | Averages predictions over multiple augmented views at inference. Free accuracy gain with no retraining required — only adds inference time.  | —             |

______________________________________________________________________

## 7. Metrics & Evaluation

### 7.1 Segmentation Metrics

| Metric                        | Status | Description                                                                                                                                 | Result                                      |
| ----------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Dice Score                    | ✅     | Measures overlap between predicted and ground-truth masks. Standard medical segmentation metric. Ranges from 0 (no overlap) to 1 (perfect). | Binary Dice, smooth=1e-6, threshold=0.5.    |
| Pixel-level F1                | ✅     | Harmonic mean of pixel precision and recall. Mathematically equivalent to Dice on binary masks but computed via torchmetrics.               | Tracked per epoch, best value checkpointed. |
| Pixel-level Accuracy          | ✅     | Fraction of correctly classified pixels. Simple but can be misleadingly high when background dominates. Tracked for completeness.           | Tracked per epoch.                          |
| IoU / Jaccard                 | 📋     | Intersection over Union — more penalising than Dice for partial overlaps. Standard benchmark metric used in most segmentation challenges.   | —                                           |
| Hausdorff Distance (95th pct) | 📋     | Measures the worst-case boundary distance (at 95th percentile). Important for clinical shape accuracy and head circumference measurement.   | —                                           |
| Average Surface Distance      | 📋     | Mean distance between predicted and ground-truth boundaries. Complementary to Dice for evaluating boundary quality specifically.            | —                                           |

### 7.2 Classification Metrics (derived)

| Metric           | Status | Description                                                                                                                                    | Result                                                 |
| ---------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Label Accuracy   | ✅     | Image-level accuracy of the derived Brain/Not-A-Brain classification. Measures how well the segmentation mask translates into a correct label. | Binary accuracy, tracked per epoch.                    |
| Label F1         | ✅     | Image-level F1 for the derived binary classification. More informative than accuracy when class distribution is imbalanced.                    | Binary F1, tracked per epoch, best value checkpointed. |
| Confusion Matrix | ✅     | 2×2 matrix showing TP/TN/FP/FN for the derived label. Provides a full picture of classification errors at test time.                           | Logged to W&B at test time (non-normalised).           |

______________________________________________________________________

## 8. Training Strategy

### 8.1 Regularisation

| Technique                             | Status | Description                                                                                                                                                                                 | Result                                                  |
| ------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Weight decay                          | ✅     | L2 penalty on model weights. Prevents weights from growing too large, reducing overfitting. Applied through the Adam optimiser.                                                             | 1e-5 via Adam.                                          |
| Early stopping                        | 📋     | Stops training when validation metric stops improving.                                                                                                                                      | patience=12, monitor=val/pixel/f1 / val/dice, mode=max. |
| Label Smoothing (for BCE)             | 📋     | Softens hard 0/1 mask targets to e.g. 0.05/0.95. Prevents over-confident predictions and can improve generalisation.                                                                        | —                                                       |
| Encoder Freezing / Gradual Unfreezing | 📋     | Freeze pretrained encoder for initial epochs (3–5), train only decoder. Then unfreeze and fine-tune end-to-end at lower LR. Prevents catastrophic forgetting and stabilizes early training. | —                                                       |

### 8.2 Training Duration & Batch

| Technique                        | Status | Description                                                                                                                           | Result                              |
| -------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 50 epochs                        | ✅     | Number of full passes through the training set. Chosen as a balance between convergence and training time.                            | max_epochs=50, min_epochs=0.        |
| Batch size 64                    | ✅     | Number of samples per gradient update. Fits comfortably in GPU memory at 192×256 resolution.                                          | batch_size=64.                      |
| Gradient accumulation (2 steps)  | 🧪     | Simulates a larger effective batch size (128) by accumulating gradients over 2 steps before updating. No extra memory needed.         | Commented out in experiment config. |
| Mixed precision (16-mixed)       | 🧪     | Uses float16 for forward/backward and float32 for weight updates. Roughly halves memory usage and speeds up training.                 | Commented out in experiment config. |
| 100+ epochs with cosine schedule | 📋     | Longer training combined with a smooth cosine LR decay. May squeeze out extra performance if the model hasn't converged at 50 epochs. | —                                   |

### 8.3 Model Selection

| Technique                         | Status | Description                                                                                                                                                                                                              | Result                                                                 |
| --------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Checkpoint on best val/pixel/f1   | ✅     | Saves model weights whenever validation pixel F1 reaches a new maximum. Ensures the deployed model is the best-performing one seen during training.                                                                      | monitor=val/pixel/f1, mode=max, save_last=True.                        |
| Checkpoint on best val/dice       | 🧪     | Use Dice as the checkpoint criterion instead of F1. Worth comparing to see if the two metrics diverge in practice.                                                                                                       | Comparable to `val/pixel/f1`, no clear winner. Keeping `val/pixel/f1`. |
| Ensemble (top-k checkpoints)      | 📋     | Average predictions from the k best checkpoints. Free accuracy gain at inference time with no additional training.                                                                                                       | —                                                                      |
| SWA (Stochastic Weight Averaging) | 📋     | Averages model weights over the last portion of training. Finds wider optima that generalise better, at no extra inference cost.                                                                                         | —                                                                      |
| Deep Supervision                  | 📋     | Adds auxiliary segmentation losses from intermediate decoder layers. Improves gradient flow to early layers, accelerates convergence, and acts as implicit regularisation. Proven in U-Net 3+ and similar architectures. | —                                                                      |
| K-Fold Cross-Validation           | 📋     | Train on k different train/val splits to obtain robust performance estimates and reduce variance from a single split. More reliable for medical datasets with limited samples.                                           | —                                                                      |

______________________________________________________________________

## 9. Post-processing

| Technique                       | Status | Description                                                                                                                                            | Result                                                                                                         |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Morphological closing / opening | 📋     | Applies morphological operators to the binary mask. Closing fills small holes, opening removes small false-positive islands.                           | —                                                                                                              |
| Connected component filtering   | 📋     | Keeps only the largest connected component. The fetal head is a single contiguous region — removing small disconnected blobs cleans up the prediction. | —                                                                                                              |
| Ellipse fitting                 | 📋     | Fits an ellipse to the predicted mask contour. Directly relevant for downstream head circumference (HC) measurement.                                   | —                                                                                                              |
| CRF (Conditional Random Field)  | ❌     | Dense CRF post-processing refines mask boundaries using image intensity cues.                                                                          | Rejected — excessive latency and complexity for a binary elliptical target. Connected-component is sufficient. |

______________________________________________________________________

## 10. Derived Classification Threshold

| Technique                        | Status | Description                                                                                                                                                                                     | Result                                             |
| -------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Fixed % pixel threshold          | ✅     | Classify as "Brain" if ≥ x % of mask pixels are positive. Simple, deterministic rule that converts segmentation output to a binary label.                                                       | threshold=0.05, applied after binarisation at 0.5. |
| Learned threshold (ROC-optimal)  | 📋     | Find the optimal pixel-percentage cutoff by maximising F1 or Youden's J on the validation set. Could improve label accuracy with minimal effort.                                                | —                                                  |
| Auxiliary classification head    | 📋     | Add a small MLP classification branch to the encoder bottleneck, trained jointly with the segmentation decoder. The model learns classification directly rather than deriving it from the mask. | —                                                  |
| Global Average Pooling → sigmoid | 📋     | Pool encoder features globally and pass through a linear+sigmoid layer. Decouples classification from mask quality — useful if mask-derived labels are noisy.                                   | —                                                  |

______________________________________________________________________

## Recommended Next Experiments (priority order)

01. **Loss: Dice + Focal** — replace BCE with Focal for hard-example mining at boundaries.
02. **Encoder freezing** — freeze encoder for 3–5 epochs, then unfreeze. Stabilises early training with pretrained weights.
03. **Scheduler: OneCycleLR** — faster convergence, potentially better final accuracy in fewer epochs.
04. **Loss: add Boundary Loss term** — distance-based loss to directly optimise boundary accuracy for HC measurement.
05. **Augmentation: ElasticTransform + Speckle Noise** — domain-specific augmentations for ultrasound.
06. **Post-processing: connected component filtering + threshold tuning** — free inference-time gains with no retraining.
07. **Encoder: EfficientNet-V2-S** — stronger features at moderate cost increase.
08. **Architecture: U-Net++ or Attention U-Net** — denser skip connections or attention gating for finer boundaries.
09. **Training: enable mixed precision** — faster iteration cycles to test more configurations.
10. **Deep Supervision** — auxiliary decoder losses for better convergence and implicit regularization.
11. **Input resolution: 224 × 320** — more spatial detail for marginal accuracy gain.

## Experiments

- test-00 - base experiment used as reference.
- test-01 - change the monitoring metric to val/dice — inconclusive: `val/dice` and `val/pixel/f1` gave similar results; kept `val/pixel/f1`.
- test-02 - Enable ImageNet normalization — slightly better avg Dice; adopted.
- test-03 - Optimiser: AdamW — plain Adam trained better for MobileNetV4; kept Adam, revisit for bigger models.
- test-04 - Test ImageNet normalization with val/pixel/f1.
- test-05 - Architecture: U-Net++ (`head_segmentation_unetpp`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-06 - Architecture: MAnet (`head_segmentation_manet`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-07 - Architecture: FPN (`head_segmentation_fpn`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-08 - Architecture: DeepLabV3+ (`head_segmentation_deeplabv3plus`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-09 - Architecture: Attention U-Net (`head_segmentation_attention_unet`, Unet + `decoder_attention_type: scse`).
- test-10 - Backbone: EfficientNet-V2-S (`head_segmentation_efficientnetv2s`, Unet + `tu-tf_efficientnetv2_s`).
- test-11 - Backbone: ConvNeXt-Tiny (`head_segmentation_convnext_tiny`, Unet + `tu-convnext_tiny`).
- test-12 - Backbone: ConvNeXt-V2-Tiny (`head_segmentation_convnextv2_tiny`, Unet + `tu-convnextv2_tiny`).
- .
- test-13 - Architecture: MAnet (`head_segmentation_manet`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-14 - Architecture: FPN (`head_segmentation_fpn`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-15 - Architecture: DeepLabV3+ (`head_segmentation_deeplabv3plus`, encoder fixed `tu-mobilenetv4_conv_small`).
- test-16 - Architecture: Attention U-Net (`head_segmentation_attention_unet`, Unet + `decoder_attention_type: scse`).
- .
- test-17 - do not use suspected images
- test-18 - reassign Axial, Brain, Other images to head
