# PLAsTiCC Novelty Detection

This project implements an autoencoder-based pipeline for novelty detection in
the PLAsTiCC astronomical transient dataset. Each object is represented by 41
numerical features extracted from its light curve and metadata.

The training set contains 14 known classes. The test set contains the same known
classes and four unseen novelty classes: `991`, `992`, `993`, and `994`. Novelty
classes are used only for evaluation.

## Method

The pipeline uses reconstruction errors to determine whether a test sample is
compatible with the known training distribution. The novelty threshold is
parametrized as:

```text
threshold = mean(training reconstruction errors)
            + k * std(training reconstruction errors)
```

Three operating points are evaluated: `k = 3`, `k = 2`, and `k = 1`. A larger
value is more conservative, while a smaller value generally increases novelty
recall at the cost of more false novelty predictions.

The project compares three experimental settings:

- **Experiment 1:** one global autoencoder trained on all known classes.
- **Experiment 2:** one autoencoder per known class, trained only on that class.
- **Experiment 3:** one autoencoder per known class, trained on all known
  classes with a custom loss that encourages class specialization.

Each experiment is trained and evaluated with both MAE- and MSE-based
reconstruction criteria.

In Experiment 1, the reconstruction error is compared with one global
threshold. In Experiments 2 and 3, each sample is reconstructed by all 14
autoencoders. The autoencoder with the lowest error determines the candidate
known class. If that error exceeds the threshold associated with the selected
autoencoder, the sample is classified as novelty; otherwise, it is assigned to
the candidate class.

## Repository Structure

```text
src/
  config.py                 Paths, classes, hyperparameters, and architectures
  data_loader.py            CSV loading, feature-label alignment, and scaling
  model.py                  Fully connected autoencoder definition
  losses.py                 MAE, MSE, and custom reconstruction losses
  utils.py                  Shared seed and reconstruction-inference utilities
  train.py                  Training functions for the three experiments
  evaluate.py               Metrics and visualization functions
  main.py                   Full training and threshold-evaluation pipeline
  threshold_sensitivity.py  Evaluation of saved models for k = 3, 2, and 1
  test_pipeline.py          Fast integration and smoke test

data/                       Local dataset files
outputs/                    Generated models, complete results, and plots
```

## Installation

From the project directory:

```cmd
pip install -r requirements.txt
```

## Usage

Run commands from the `src` directory:

```cmd
cd /d C:\path\to\Big-Data\src
```

### Complete Pipeline

```cmd
python main.py
```

`main.py` performs the complete workflow:

1. loads and standardizes the datasets;
2. trains all Experiment 1, 2, and 3 models with MAE and MSE;
3. saves the trained model weights and training histories;
4. evaluates the saved models with `k = 3`, `k = 2`, and `k = 1`;
5. saves complete JSON results and plots for every configuration.

The threshold comparison is executed by the same functions exposed through
`threshold_sensitivity.py`.

### Threshold Evaluation Without Retraining

```cmd
python threshold_sensitivity.py
```

Use this script when the models have already been trained. It reloads the model
weights from `outputs/models/`, recomputes reconstruction errors and thresholds,
and regenerates the results for all three values of `k` without training the
autoencoders again.

Training histories are loaded from
`outputs/models/training_metadata.json`, ensuring that regenerated JSON files
remain complete. The script overwrites the corresponding files under the
threshold-specific result and plot directories.

## Outputs

```text
outputs/
  models/
    *.pt                    Trained PyTorch model weights
    training_metadata.json Training histories and final-loss statistics
  results/
    sigma_1/                Complete JSON results for mean + 1 std
    sigma_2/                Complete JSON results for mean + 2 std
    sigma_3/                Complete JSON results for mean + 3 std
  plots/
    sigma_1/                Plots for mean + 1 std
    sigma_2/                Plots for mean + 2 std
    sigma_3/                Plots for mean + 3 std
```

Each complete JSON result contains the relevant thresholds, predictions,
reconstruction errors, true labels, summary metrics, and training history.
Class-specific experiments also include the best-reconstructing class for every
test sample.

Generated visual outputs include:

- Experiment 1 reconstruction-error distributions;
- Experiment 2 and 3 train/test reconstruction-error heatmaps;
- minimum reconstruction-error distributions for Experiments 2 and 3.

The reported metrics are novelty recall, false novelty rate, per-class novelty
recall, and known-class accuracy for Experiments 2 and 3.
