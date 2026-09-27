# ACDM-DSC: Zero-Shot Compound Fault Diagnosis

This directory contains the cleaned release package for the paper **A Zero-Shot Method for Cross-Condition Compound Fault Diagnosis Based on an Attribute-Conditioned Diffusion Model and Distribution Shift Calibration**.

The release preserves the original model architecture, labels, loss functions, default hyperparameters, and data-processing behavior. Only engineering changes were added: configurable paths, dependency documentation, command-line entry points, and a minimal smoke test.

## Mapping to the paper

- `data_generate/data_set.py`: raw vibration loading, sliding windows, and time/frequency/angular-domain preprocessing.
- `data_utils.py`: angular-frequency processing and data caching.
- `ddpm_model.py`: one-dimensional conditional diffusion U-Net.
- `train_ddpm.py`: ACDM training, classifier-free guidance, and mixed-condition sampling.
- `dg_model.py`: diagnostic feature extractor and DSC geometric constraints.
- `train_dg_all.py`: joint real/generated-sample training, cross-condition evaluation, and result aggregation.

DSC is implemented indirectly according to the bias-consistency assumption in the paper. Real and generated samples from known classes jointly participate in the intra-class compactness constraint, while generated compound-fault samples are indirectly calibrated through the inter-class separation constraint.

## Environment

Python 3.10 or 3.11 is recommended. Install a PyTorch build matching your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The original datasets, checkpoints, and experimental result files are not included. Prepare the data directory before running the scripts.

## Data layout

The training scripts expect one working-condition directory per subdirectory under the dataset root, for example:

```text
/path/to/bearing_313lb/
├── 950rpm/
├── 1000rpm/
├── 1050rpm/
└── 1100rpm/
```

The raw file format and naming rules are selected by `DatasetConfig` and its dataset-specific loader. Dataset names and default working conditions are defined in the `Config` class in `train_dg_all.py`.

## Running the code

### 1. Run the smoke test

```bash
python scripts/smoke_test.py
```

This checks module imports, model construction, and one forward pass. It does not read data or train a model.

### 2. Train ACDM and generate samples

Example for the `313lb` dataset:

```bash
python train_ddpm.py \
  --data-root /path/to/bearing_313lb \
  --results-root ./results_diffusion_313lb \
  --conditions 950rpm 1000rpm 1050rpm 1100rpm
```

For each working condition, the script trains one diffusion model and writes files such as:

```text
results_diffusion_313lb/<condition>/model_final.pth
results_diffusion_313lb/<condition>/test_results_500_GW2.0/
├── single_samples/
└── compound_samples/
```

### 3. Train the diagnostic model and evaluate DSC

```bash
python train_dg_all.py \
  --dataset 313lb \
  --data-root /path/to/bearing_313lb \
  --results-root ./results_diffusion_313lb \
  --distance-metric cosine \
  --runs 5
```

The diagnostic script reads the single-fault and compound-fault samples generated in the previous step and writes model checkpoints, per-condition Excel results, and summary workbooks.

## Reproducibility notes

1. The default training epochs, diffusion steps, class indices, and random seeds are preserved from the original implementation.
2. Before ACDM training, verify that `--data-root/<condition>` exists and contains files supported by the selected dataset loader.
3. DSC training requires the `single_samples` and `compound_samples` outputs from the ACDM stage.
4. Run commands from this directory, or add this directory to `PYTHONPATH`.
5. CUDA is optional. CPU execution is supported automatically but will be substantially slower for the full experiments.

## Repository layout

```text
.
├── data_generate/
├── data_utils.py
├── ddpm_model.py
├── dg_model.py
├── train_ddpm.py
├── train_dg_all.py
├── requirements.txt
└── scripts/smoke_test.py
```

## Citation

If this code is useful in your research, please cite the associated paper. This release preserves the paper's indirect DSC procedure and does not replace it with an additional explicit MMD, CORAL, or other distribution-matching loss.
