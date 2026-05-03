"""
Training driver for tree-based baselines: Random Forest, XGBoost, LightGBM.

Usage:
    python train_gbm.py --model xgb --soil with --config config/xgb.yaml
    python train_gbm.py --model lgbm --soil without --config config/lgbm.yaml
    python train_gbm.py --model rf --soil with --config config/rf.yaml

Follows the shared protocol from paper_revision/action_items.md §2.0 / §2.4 / §2.5:
- Optuna TPE HPO (200 trials), minimizing validation MSE (L2 only)
- 5-seed final evaluation with best HPO params
- Artifacts: .joblib model, metadata YAML, test results CSV, evaluation plots
"""

import os
import sys

# macOS: XGBoost (libomp) + NumPy/Accelerate or MKL OpenMP in the same process
# often segfaults unless thread pools are capped and duplicate OpenMP is allowed.
if sys.platform == "darwin":
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_k, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import optuna
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Allow imports from sibling packages.
# training_utils.py has a top-level `from models import CNN_RNN`, so src/rnn
# must also be on sys.path for that import to resolve.
_project_src = os.path.join(os.path.dirname(__file__), '..')
_rnn_dir = os.path.join(_project_src, 'rnn')
for _p in (_project_src, _rnn_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rnn.training_utils import calculate_metrics, create_evaluation_plots, save_results

try:
    from analysis.plot_style import apply_style
    apply_style()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

METADATA_COLS = ['sown_area', 'total_area', 'district', 'year']
TARGET_COL = 'yield_value'


def load_data(config, soil_mode, config_dir):
    """Load train/val/test DataFrames and split into features + target."""
    base = config['data']['base_path']
    if not os.path.isabs(base):
        base = os.path.normpath(os.path.join(config_dir, base))
    train_df = pd.read_csv(os.path.join(base, config['data']['train_file']))
    val_df = pd.read_csv(os.path.join(base, config['data']['val_file']))
    test_df = pd.read_csv(os.path.join(base, config['data']['test_file']))

    feature_cols = _feature_columns(train_df, soil_mode)

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values

    meta = {
        'feature_cols': feature_cols,
        'test_districts': test_df['district'].tolist(),
        'test_years': test_df['year'].astype(int).tolist(),
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


def _feature_columns(df, soil_mode):
    temporal = sorted([c for c in df.columns if c.startswith('temporal_')])
    soil = sorted([c for c in df.columns if c.startswith('soil_')])
    cols = temporal + ['past_yield']
    if soil_mode == 'with':
        cols += soil
    return cols


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _build_xgb(params, seed):
    from xgboost import XGBRegressor
    return XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _build_lgbm(params, seed):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        objective='regression',
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        min_split_gain=params['min_split_gain'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


def _build_rf(params, seed):
    max_depth = params['max_depth']
    max_features = params['max_features']
    if isinstance(max_features, (int, float)) and max_features == int(max_features):
        max_features = float(max_features)
    return RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=max_depth,
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=max_features,
        random_state=seed,
        n_jobs=1,
    )


MODEL_BUILDERS = {
    'xgb': _build_xgb,
    'lgbm': _build_lgbm,
    'rf': _build_rf,
}


# ---------------------------------------------------------------------------
# HPO objective
# ---------------------------------------------------------------------------

def _sample_params(trial, model_key, search_space):
    """Sample hyperparameters from the Optuna search space config."""
    params = {}
    for name, spec in search_space.items():
        ptype = spec['type']
        if ptype == 'int':
            params[name] = trial.suggest_int(name, spec['low'], spec['high'])
        elif ptype == 'float':
            params[name] = trial.suggest_float(name, spec['low'], spec['high'])
        elif ptype == 'loguniform':
            params[name] = trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
        elif ptype == 'categorical':
            params[name] = trial.suggest_categorical(name, spec['choices'])
        else:
            raise ValueError(f"Unknown param type: {ptype}")
    return params


def objective(trial, model_key, config, X_train, y_train, X_val, y_val):
    search_space = config['optimization']['hparam_search_space']
    params = _sample_params(trial, model_key, search_space)
    seed = config['reproducibility']['seed']
    builder = MODEL_BUILDERS[model_key]
    model = builder(params, seed)

    early_rounds = config.get('training', {}).get('early_stopping_rounds', 50)

    if model_key == 'xgb':
        model.set_params(early_stopping_rounds=early_rounds)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
    elif model_key == 'lgbm':
        from lightgbm import early_stopping as lgbm_early_stopping, log_evaluation
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mse',
            callbacks=[lgbm_early_stopping(early_rounds, verbose=False), log_evaluation(period=-1)],
        )
        preds = model.predict(X_val)
    else:  # rf
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

    val_mse = mean_squared_error(y_val, preds)
    return val_mse


def run_hpo(model_key, config, X_train, y_train, X_val, y_val, soil_mode):
    """Run Optuna HPO and return best params dict."""
    baseline_name = f"{model_key}_{soil_mode}_soil"
    hpo_base = config['optimization']['hpo_dir']
    hpo_dir = os.path.join(hpo_base, baseline_name)
    os.makedirs(hpo_dir, exist_ok=True)

    study_name = f"{config['optimization']['study_name']}_{soil_mode}_soil"
    storage_path = f"sqlite:///{os.path.join(hpo_dir, f'{study_name}.db')}"

    config_yaml_path = os.path.join(hpo_dir, f'{study_name}_config.yaml')
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction='minimize',
        load_if_exists=True,
    )

    obj = partial(objective, model_key=model_key, config=config,
                  X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    n_trials = config['optimization']['n_trials']
    timeout = config['optimization'].get('timeout_hours', 4) * 3600

    study.optimize(obj, n_trials=n_trials, timeout=timeout)

    print(f"\nHPO complete. Best val MSE: {study.best_trial.value:.2f}")
    print(f"Best params: {study.best_trial.params}")

    best_params_path = os.path.join(hpo_dir, f'{study_name}_best_params.yaml')
    with open(best_params_path, 'w') as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False, sort_keys=False)

    return study.best_trial.params


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def train_and_evaluate_seed(model_key, best_params, config, seed,
                            X_train, y_train, X_val, y_val, X_test, y_test, meta,
                            soil_mode):
    """Train with best params on train set, evaluate on test set, save artifacts.
    Returns (test_metrics, model_save_path, model)."""
    set_random_seed(seed)
    builder = MODEL_BUILDERS[model_key]
    model = builder(best_params, seed)

    early_rounds = config.get('training', {}).get('early_stopping_rounds', 50)

    if model_key == 'xgb':
        model.set_params(early_stopping_rounds=early_rounds)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_key == 'lgbm':
        from lightgbm import early_stopping as lgbm_early_stopping, log_evaluation
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mse',
            callbacks=[lgbm_early_stopping(early_rounds, verbose=False), log_evaluation(period=-1)],
        )
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = calculate_metrics(y_test, preds)

    baseline_name = f"{model_key}_{soil_mode}_soil"
    output_base = config['output']['model_save_dir']
    output_dir = os.path.join(output_base, baseline_name)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{timestamp}"

    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    results_path = os.path.join(output_dir, f"{model_name}_test_results.csv")
    save_results(y_test, preds, meta['test_districts'], meta['test_years'], metrics, results_path)

    plot_path = os.path.join(output_dir, f"{model_name}_evaluation_plots.png")
    create_evaluation_plots(y_test, preds, plot_path)

    return metrics, model_path, model_name, output_dir


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python for clean YAML output."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def multi_seed_evaluation(model_key, best_params, config,
                          X_train, y_train, X_val, y_val, X_test, y_test,
                          meta, soil_mode):
    """Run 5-seed evaluation, keep best model, save aggregated metadata."""
    base_seed = config['reproducibility']['seed']
    num_runs = config['reproducibility']['num_runs']
    np.random.seed(base_seed)
    seeds = [int(s) for s in np.random.randint(0, 10000, size=num_runs)]

    print(f"\n{'='*60}")
    print(f"Running {num_runs} seeds: {seeds}")
    print(f"{'='*60}")

    all_metrics = []
    all_paths = []
    best_r2 = -float('inf')
    best_info = {}

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {run_idx+1}/{num_runs}: {seed} ---")
        metrics, model_path, model_name, output_dir = train_and_evaluate_seed(
            model_key, best_params, config, int(seed),
            X_train, y_train, X_val, y_val, X_test, y_test, meta, soil_mode,
        )
        all_metrics.append(metrics)
        all_paths.append((model_path, model_name, output_dir))

        if metrics['r2_score'] > best_r2:
            best_r2 = metrics['r2_score']
            best_info = {
                'model_path': model_path,
                'model_name': model_name,
                'output_dir': output_dir,
                'seed': int(seed),
                'test_metrics': metrics,
            }

    # Clean up non-best artifacts
    for model_path, model_name, output_dir in all_paths:
        if model_path != best_info['model_path']:
            for path in [
                model_path,
                os.path.join(output_dir, f"{model_name}_test_results.csv"),
                os.path.join(output_dir, f"{model_name}_evaluation_plots.png"),
            ]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

    # Aggregate metrics
    metrics_df = pd.DataFrame(all_metrics)
    mean_metrics = metrics_df.mean().to_dict()
    std_metrics = metrics_df.std().to_dict()

    # Save metadata YAML
    metadata = _convert_numpy({
        'model_type': model_key,
        'soil_mode': soil_mode,
        'best_hpo_params': best_params,
        'random_seed': best_info['seed'],
        'seeds': seeds,
        'test_metrics': best_info['test_metrics'],
        'mean_test_metrics': mean_metrics,
        'std_test_metrics': std_metrics,
        'all_test_metrics': all_metrics,
        'config': config,
    })

    metadata_path = os.path.join(
        best_info['output_dir'],
        f"{best_info['model_name']}_metadata.yaml",
    )
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Best seed: {best_info['seed']}  |  R² = {best_r2:.4f}")
    print(f"Mean test metrics (over {num_runs} seeds):")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.4f} ± {std_metrics[k]:.4f}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train tree-based baseline models')
    parser.add_argument('--model', required=True, choices=['rf', 'xgb', 'lgbm'],
                        help='Model type')
    parser.add_argument('--soil', required=True, choices=['with', 'without'],
                        help='Include soil features or not')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config file')
    parser.add_argument('--skip-hpo', action='store_true',
                        help='Skip HPO and use params from existing best_params YAML')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override number of HPO trials from config')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.n_trials is not None:
        config['optimization']['n_trials'] = args.n_trials

    # Resolve relative paths in config against the config file's directory
    for key in ('hpo_dir',):
        val = config.get('optimization', {}).get(key)
        if val and not os.path.isabs(val):
            config['optimization'][key] = os.path.normpath(os.path.join(config_dir, val))
    out_dir = config.get('output', {}).get('model_save_dir')
    if out_dir and not os.path.isabs(out_dir):
        config['output']['model_save_dir'] = os.path.normpath(os.path.join(config_dir, out_dir))

    print(f"Model: {args.model}  |  Soil: {args.soil}")
    print(f"Loading data...")

    X_train, y_train, X_val, y_val, X_test, y_test, meta = load_data(config, args.soil, config_dir)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Features: {len(meta['feature_cols'])}")

    if args.skip_hpo:
        baseline_name = f"{args.model}_{args.soil}_soil"
        study_name = f"{config['optimization']['study_name']}_{args.soil}_soil"
        bp_path = os.path.join(
            config['optimization']['hpo_dir'], baseline_name,
            f"{study_name}_best_params.yaml",
        )
        print(f"Loading best params from: {bp_path}")
        with open(bp_path, 'r') as f:
            best_params = yaml.safe_load(f)
    else:
        best_params = run_hpo(
            args.model, config, X_train, y_train, X_val, y_val, args.soil,
        )

    multi_seed_evaluation(
        args.model, best_params, config,
        X_train, y_train, X_val, y_val, X_test, y_test,
        meta, args.soil,
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
