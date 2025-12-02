# Reddit Thread Prediction Models

A comprehensive machine learning pipeline for predicting Reddit thread outcomes using LightGBM classifiers. This repository contains code for two-stage modeling: binary classification of thread initiation (started vs. stalled) and multiclass prediction of thread size categories.

## Overview

This project analyzes Reddit discussion threads to predict:
1. **Stage 1**: Whether a thread will start (receive at least one comment) or stall
2. **Stage 2**: The size category of a thread (stalled, small, medium, large)

The pipeline includes feature engineering, TF-IDF text vectorization with SVD dimensionality reduction, hyperparameter tuning via Optuna, and comprehensive model evaluation with SHAP interpretability analysis.

## Repository Structure
```
.
└── 0_Preprocessing
    ├── 1_construct_features.py          # Feature engineering from raw Reddit data
    ├── 2_tf_idf_analysis.py             # TF-IDF vectorizer and SVD tuning
    └── 3_model_data.py                  # Final train/test split preparation
└── 1_Thread_start
    ├── 1_feature_baselines.py           # Stage 1 baseline feature evaluation
    ├── 2_tuning.py                      # Stage 1 class weight and threshold tuning
    ├── 3_hyperparameter_tuning.py       # Stage 1 tree hyperparameter optimization
    └── 4_run_tuned_model.py             # Stage 1 final model evaluation
└── 2_Thread_size
    ├── 1_feature_baselines.py           # Stage 2 baseline feature evaluation
    ├── 2_tuning.py                      # Stage 2 class weight and threshold tuning
    ├── 3_hyperparameter_tuning.py       # Stage 2 tree hyperparameter optimization
    └── 4_run_tuned_model.py             # Stage 2 final model evaluation
└── 3_Graphs_Tables
    └── make_outputs.py                  # Publication-ready figures and tables
```

## Installation

### Requirements

- Python 3.8+
- Key dependencies:
```bash
  pip install pandas numpy scikit-learn lightgbm optuna shap
  pip install matplotlib seaborn joblib nltk pillow
```

### NLTK Setup

Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
           nltk.download('averaged_perceptron_tagger'); \
           nltk.download('averaged_perceptron_tagger_eng'); \
           nltk.download('stopwords')"
```

## Usage

### 0. Preprocessing Pipeline

**Step 1: Feature Construction**
```bash
python 1_construct_features.py \
    --subreddit conspiracy \
    --outdir ./output/features \
    --comments ./data/conspiracy_comments.parquet \
    --threads ./data/conspiracy_threads.parquet
```

**Step 2: TF-IDF and SVD Optimization**
```bash
python 2_tf_idf_analysis.py \
    --subreddit conspiracy \
    --outdir ./output/tfidf \
    --comments ./output/features/conspiracy_comments_extra_feats.parquet \
    --threads ./output/features/conspiracy_threads_extra_feats.parquet \
    --rs 42 \
    --trials 100 \
    --splits 5
```

**Step 3: Train/Test Split Preparation**
```bash
python 3_model_data.py \
    --subreddit conspiracy \
    --outdir ./output/model_data \
    --train ./output/tfidf/conspiracy_svd_enriched_train_data.parquet \
    --test ./output/tfidf/conspiracy_svd_enriched_test_data.parquet \
    --y-col thread_size \
    --corr 0.5 \
    --rs 42
```

### Stage 1: Thread Start Prediction (Binary Classification)

**Step 1.1: Feature Baseline Evaluation**
```bash
python 1_feature_baselines.py \
    --subreddit conspiracy \
    --outdir ./output/stage1/baselines \
    --train_X ./output/model_data/conspiracy_train_X.parquet \
    --train_y ./output/model_data/conspiracy_train_Y.parquet \
    --y-col log_thread_size \
    --feats 30 \
    --splits 5 \
    --n-bs 1000 \
    --rs 42
```

**Step 1.2: Class Weight and Threshold Tuning**
```bash
python 2_tuning.py \
    --subreddit conspiracy \
    --outdir ./output/stage1/tuning \
    --train_X ./output/model_data/conspiracy_train_X.parquet \
    --train_y ./output/model_data/conspiracy_train_Y.parquet \
    --scorer MCC \
    --feats 20 \
    --trials 300 \
    --splits 5 \
    --rs 42
```

**Step 1.3: Tree Hyperparameter Optimization**
```bash
python 3_hyperparameter_tuning.py \
    --subreddit conspiracy \
    --outdir ./output/stage1/hyperparams \
    --train_X ./output/model_data/conspiracy_train_X.parquet \
    --train_y ./output/model_data/conspiracy_train_Y.parquet \
    --params ./output/stage1/tuning/tuned_params.jl \
    --scorer MCC \
    --trials 300 \
    --splits 5 \
    --rs 42
```

**Step 1.4: Final Model Evaluation**
```bash
python 4_run_tuned_model.py \
    --subreddit conspiracy \
    --outdir ./output/stage1/final \
    --train_X ./output/model_data/conspiracy_train_X.parquet \
    --test_X ./output/model_data/conspiracy_test_X.parquet \
    --train_y ./output/model_data/conspiracy_train_Y.parquet \
    --test_y ./output/model_data/conspiracy_test_Y.parquet \
    --params ./output/stage1/hyperparams/params_post_hyperparam_tuning.jl \
    --tfidf ./output/tfidf/conspiracy_optuna_tfidf_vectorizer.jl \
    --svd ./output/tfidf/conspiracy_optuna_svd_model.jl \
    --scorer MCC \
    --splits 5 \
    --n-bs 1000 \
    --rs 42
```

### Stage 2: Thread Size Prediction (Multiclass Classification)

The Stage 2 pipeline follows the same structure as Stage 1:

**Step 2.1: Feature Baseline Evaluation**
```bash
python 1_feature_baselines.py \
    --subreddit conspiracy \
    --outdir ./output/stage2/baselines \
    --train_X ./output/model_data/conspiracy_train_X.parquet \
    --train_y ./output/model_data/conspiracy_train_Y.parquet \
    --classes 4 \
    --feats 30 \
    --splits 5 \
    --n-bs 1000 \
    --rs 42
```

**Steps 2.2-2.4**: Follow the same pattern as Stage 1, adding `--classes 4` (or `--classes 3`) to specify the number of size categories.

### Generating Publication Outputs

After running both stages for all subreddits, generate publication-ready figures and tables:

**Stage 1 Outputs**
```bash
python make_outputs.py \
    --stage 1 \
    --root ./output/stage1/final \
    --selected-models ./config/selected_models_stage1.csv \
    --outdir ./publication/stage1
```

**Stage 2 Outputs**
```bash
python make_outputs.py \
    --stage 2 \
    --n-classes 4 \
    --root ./output/stage2/final \
    --selected-models ./config/selected_models_stage2.csv \
    --outdir ./publication/stage2
```

The `selected_models.csv` file should specify which feature count to use for each subreddit:
```csv
subreddit,n_feats
conspiracy,15
crypto,20
politics,18
```

## Key Features

### Feature Engineering
- **Text features**: TF-IDF with SVD dimensionality reduction, word statistics, POS ratios
- **Structural features**: Thread depth, reply counts, sentiment metrics
- **Temporal features**: Hour of day, day of week
- **Domain features**: Image/video detection, external link classification
- **Author features**: Frequency encoding for authors and domains

### Model Architecture
- **Classifier**: LightGBM with class-balanced weights
- **Calibration**: Isotonic or sigmoid probability calibration
- **Threshold optimization**: Per-class probability thresholds tuned via L-BFGS-B
- **Cross-validation**: Stratified K-fold with out-of-fold predictions

### Hyperparameter Tuning
- **Optimizer**: Optuna with Tree-structured Parzen Estimator (TPE) sampler
- **Search space**: Class weights, tree structure, regularization, learning rates
- **Metrics**: Matthews Correlation Coefficient (MCC), F1-score, balanced accuracy
- **Validation**: Bootstrap confidence intervals (n=1000) for all metrics

### Interpretability
- **SHAP values**: TreeExplainer for feature importance and interaction analysis
- **Confusion matrices**: Bootstrap confidence intervals for all cells
- **Feature importance**: Combined split-based and gain-based rankings
- **Per-feature analysis**: SHAP scatter plots for top features

## Output Files

Each stage produces:
- **Trained models**: Serialized LightGBM classifiers and calibrated wrappers (`.jl`)
- **Predictions**: Out-of-fold and test set probabilities (`.parquet`)
- **Metrics**: Performance scores with bootstrap CIs (`.xlsx`)
- **SHAP analyses**: Feature importance rankings and visualizations (`.png`, `.eps`)
- **Confusion matrices**: With bootstrap uncertainty estimates (`.xlsx`, `.png`)
- **Hyperparameters**: Complete tuning logs and convergence plots (`.xlsx`, `.png`)

## Debug Mode

All scripts support `--debug` mode for rapid iteration:
```bash
python 1_feature_baselines.py --debug \
    --subreddit conspiracy \
    --outdir ./debug \
    --train_X ./data/train_X.parquet \
    --train_y ./data/train_Y.parquet
```

Debug mode reduces:
- CV splits: 5 → 2
- Optuna trials: 100-300 → 10
- Bootstrap iterations: 1000 → 20
- Feature counts: 30 → 10

## Reproducibility

All random operations are seeded via `--rs` (default: 42):
- Cross-validation splits
- Optuna sampler
- LightGBM model training
- Bootstrap resampling

Each output includes:
- Complete command-line arguments
- Library versions (Python, pandas, numpy, LightGBM, scikit-learn, Optuna, SHAP)
- Runtime information
- Random seed values

## Citation

If you use this code, please cite:
```bibtex
@software{reddit_thread_prediction,
  title = {Reddit Thread Prediction Models},
  author = {Cara Lynch},
  year = {2025},
  url = {https://github.com/caralynch/thread-size}
}
```

## License

[Specify your license here, e.g., MIT, Apache 2.0]

## Support

For issues or questions:
- Open a GitHub issue
- Email: ucabcpl@ucl.ac.uk

## Acknowledgments

This project uses:
- [LightGBM](https://github.com/microsoft/LightGBM) for gradient boosting
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [SHAP](https://github.com/slundberg/shap) for model interpretability
- [scikit-learn](https://scikit-learn.org/) for preprocessing and evaluation