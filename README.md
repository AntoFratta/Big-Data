# PLAsTiCC Novelty Detection

This project reproduces the autoencoder-based novelty detection experiments described in the reference thesis on the PLAsTiCC dataset. The main extension is the introduction of a threshold-based decision rule using:

```text
threshold = mean(training reconstruction error) + k * std(training reconstruction error)
```

The implemented comparison evaluates `k = 3`, `k = 2`, and `k = 1`.

## Project Context

The dataset, feature set, autoencoder architectures, and loss functions follow the original thesis setup. The difference introduced here is the final decision criterion used to identify novelty samples.

For Experiment 1, a single global autoencoder is trained on all known classes. A test sample is classified as novelty if its reconstruction error is above the global threshold.

For Experiments 2 and 3, one autoencoder is associated with each known class. For each test sample, the reconstruction error is computed across all class-specific autoencoders. The sample is assigned to the class of the autoencoder with the lowest reconstruction error, unless that minimum error exceeds the threshold associated with that autoencoder; in that case, it is classified as novelty.

## Experiments

- **Experiment 1**: one global autoencoder trained on all known classes.
- **Experiment 2**: one autoencoder per known class, trained only on samples from that class.
- **Experiment 3**: one autoencoder per known class, trained on all known classes using the custom loss from the thesis.

Each experiment is evaluated with both MAE and MSE reconstruction losses.

## Repository Structure

```text
src/
  config.py                 Configuration, paths, classes, architectures
  data_loader.py            Dataset loading, merging, and standardization
  model.py                  Autoencoder definition
  losses.py                 Reconstruction and custom losses
  train.py                  Training and threshold utilities
  evaluate.py               Metrics and plots
  main.py                   Full training + threshold comparison pipeline
  threshold_sensitivity.py  Re-evaluation of saved models with k = 3, 2, 1
  test_pipeline.py          Fast integration test

data/                       Local dataset files, ignored by git
outputs/                    Generated models, results, and plots, ignored by git
```

## Data Layout

The expected local data structure is:

```text
data/
  train/
    dataset_augment_zeros.csv
    y_dataset_augment.csv
  test/
    dataset_test_zeros.csv
    y_dataset_test.csv
```

The CSV files are not versioned because of their size.

## Usage

Install the required packages:

```bash
pip install -r requirements.txt
```

Run a fast integration test:

```bash
cd src
python test_pipeline.py
```

Run the full pipeline from scratch:

```bash
cd src
python main.py
```

This trains all models and then evaluates the three threshold settings.

If the models have already been trained, run only the threshold comparison:

```bash
cd src
python threshold_sensitivity.py
```

This reloads the models from `outputs/models/` and regenerates the results and plots for `sigma_3`, `sigma_2`, and `sigma_1`.

## Outputs

Generated files are saved under:

```text
outputs/models/             Trained PyTorch models
outputs/results/sigma_*/    Full JSON results for each threshold setting
outputs/plots/sigma_*/      Heatmaps and reconstruction-error distributions
```

The JSON results include the threshold values, summary metrics, predictions, reconstruction errors, true labels, and training histories where available.

The main metrics are:

- novelty recall;
- false novelty rate;
- per-class novelty recall;
- known-class accuracy for Experiments 2 and 3.

## Notes

The threshold factor controls the trade-off between novelty detection and false positives. A larger factor, such as `k = 3`, is more conservative and usually produces fewer false novelty predictions, while smaller factors increase novelty recall at the cost of more known samples being flagged as novelty.
