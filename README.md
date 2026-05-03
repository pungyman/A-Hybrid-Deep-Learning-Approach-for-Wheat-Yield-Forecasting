# Winter Wheat Yield Forecasting in India

This repository contains the public implementation for a multimodal winter wheat yield forecasting model over the Indian Wheat Belt. The main model combines a soil-depth CNN branch with a temporal BiLSTM and self-attention branch to integrate soil properties, meteorological variables, vegetation indices, and lagged yield.

The GitHub repository is intentionally code-only. Datasets, trained checkpoints, hyperparameter studies, and generated paper figures are kept outside GitHub and can be synchronized from the companion Hugging Face dataset repository when available.

## Repository Structure

```text
.
├── data/
│   ├── soilgrids/        # SoilGrids acquisition and wrangling scripts
│   ├── weather_data/     # NASA POWER weather acquisition and QA scripts
│   └── yield_data/       # Yield-data merge/wrangling scripts
├── src/
│   ├── analysis/         # Paper-result analysis and plotting scripts
│   ├── baselines/        # CNN, Transformer, parallel CNN-LSTM, and GBM baselines
│   ├── preprocessing/    # Monthly aggregation and final dataset creation
│   └── rnn/              # Main CNN-BiLSTM-attention model, training, HPO, SHAP
├── hf_sync.sh            # Sync datasets, checkpoints, and HPO artifacts
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/pungyman/crop_yield_prediction.git
cd crop_yield_prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The training code supports CPU, CUDA, and Apple Metal where the installed PyTorch build supports them. Full model training and SHAP attribution are substantially faster on a CUDA GPU.

## Data And Model Artifacts

Large artifacts are not stored in GitHub. The helper script expects the Hugging Face CLI to be authenticated locally:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
./hf_sync.sh pull
```

Useful variants:

```bash
./hf_sync.sh pull-data
./hf_sync.sh pull
./hf_sync.sh push-models
```

The script syncs these local artifact directories when present:

- `data/datasets/final/` for sequence datasets used by the deep learning models.
- `data/datasets/rf_flat/` for flattened tabular datasets used by tree baselines.
- `src/rnn/saved_models/` for trained checkpoints and metadata.
- `experiments/hpo/` for Optuna studies and selected hyperparameters.

## Reproduction Workflow

After artifacts are available locally, run the main model from `src/rnn`:

```bash
cd src/rnn
python train_model.py --config training_config.yaml
python run_inference_all_splits.py --config training_config.yaml --model-dir saved_models/cnn_bilstm_self_attn
python run_shap_analysis.py --config shap_config.yaml
```

The preprocessing scripts can be rerun if the raw inputs are available locally:

```bash
python src/preprocessing/get_monthly_features.py
python src/preprocessing/create_dataset.py
```

Several scripts use relative paths that assume they are launched from their source directory. If a path is not found, run the command from the directory containing that script or adjust the YAML/path constants for your local artifact layout.

## Baselines

Deep learning baselines live under `src/baselines/`:

```bash
cd src/baselines
python train_cnn1d.py --config config/cnn1d_with_soil.yaml
python train_cnn1d.py --config config/cnn1d_without_soil.yaml
python train_transformer.py --config config/transformer_with_soil.yaml
python train_parallel.py --config config/parallel_with_soil.yaml
```

Tree-based baselines use the flattened dataset:

```bash
cd src/baselines
python train_gbm.py --model xgb --soil with --config config/xgb.yaml
python train_gbm.py --model lgbm --soil without --config config/lgbm.yaml
python train_gbm.py --model rf --soil with --config config/rf.yaml
```

## Analysis Scripts

Analysis and plotting utilities are in `src/analysis/`. They read saved model metadata, prediction CSVs, and generated datasets, then write outputs to ignored local artifact directories.

Common entry points include:

```bash
python src/analysis/build_baseline_table.py
python src/analysis/feature_correlation.py
python src/analysis/scatter_grid.py
python src/analysis/persistence_baseline.py
python src/analysis/yield_histogram.py
```

## Data Sources

This repository provides code for assembling the modeling dataset, but it does not redistribute all raw inputs. Users are responsible for complying with the terms of each data provider.

- Weather features are derived from NASA POWER.
- Soil properties are derived from SoilGrids/ISRIC.
- Remote-sensing vegetation indices are derived from MODIS/GEE workflows.
- District boundaries and yield statistics must be supplied according to the relevant source licenses and redistribution terms.

## Citation

If you use this implementation, please cite the associated paper once the final citation is available.
