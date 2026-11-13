# APEX-MRE

This repository contains the **peer-review version** of the APEX-MRE implementation. It provides the following core components.

- **Endpoint-adapted classifier** in `apex_mre/classifier.py`. Injects visual endpoint features at the head and tail markers and applies seven-view residual gating.
- **APCI** in `apex_mre/intervention.py`. Preserves the relation arguments and constructs paired views by replacing unprotected context slots.
- **SE-ARA** in `apex_mre/exposure.py` and `apex_mre/objective.py`. Builds exposure groups from training data and updates group weights according to group risk.
- **D-JOP** in `apex_mre/objective.py` and `apex_mre/training.py`. Combines classification and prediction consistency while updating group weights from clean classification losses.

## Run the core example

Use Python 3.9 or newer and run these commands from the repository root.

    python -m pip install -r requirements.txt
    python run_core.py

The CPU example checks argument preservation, exposure grouping, the D-JOP objective and its gradients, and unit gate weights at initialization.

## Classifier and training

`UnifiedFocal` wraps an initialized FocalMRE classifier with BERT and CLIP encoders. It injects visual endpoint features at the head and tail markers and adds the gated view contributions to the base logits. [FocalMRE source](https://github.com/NJUNLP/FocalMRE)

Fit `ExposureGroups` on the training split. Each batch uses the corresponding training group IDs and shares one integer `source_code` across records with the same decoded RGB source. Call `risk.start_epoch()` before the first batch, use `training_loss` to compute the loss for each batch, and call `risk.finish_epoch()` after the complete training split.

For the MNRE exposure-only setting, use a classifier that returns logits and `None` for the gate weights, and pass `apci=False`.

## Benchmark data

The datasets are available from their original releases. [MNRE](https://github.com/thecharm/MNRE), [MORE](https://github.com/NJUNLP/MORE), [UMRE](https://github.com/Nikol-coder/REMOTE).

After extracting MORE, run the following command with the paths to the released training split and images.

    python -m apex_mre.exposure --train data/MORE/txt/train.txt --image-dir path/to/MORE/images

The command reports the number of examples in each exposure group.

## Full code release

The complete codebase, including the end-to-end training and evaluation scripts and experiment configurations, will be released in this repository upon acceptance of the manuscript.
