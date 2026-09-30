# Apartment Rent Price Prediction

Laboratory Work 1 for the **Computational Intelligence and Decision Making** course at KTU.

This project predicts monthly apartment rental prices and compares two regression algorithms implemented from scratch with their scikit-learn equivalents and a Random Forest. It includes exploratory data analysis, preprocessing, validation experiments, independent test evaluation, and plots generated from saved experiment results.

## Dataset

The pipeline uses the Apartments for Rent Classified 100K dataset. The extracted input file must be available at:

```text
data/apartments_for_rent_classified_100K.csv
```

The CSV is read with a semicolon separator and Latin-1 encoding. The supplied file contains 99,492 rows and 22 columns. The target is `price`, measured in USD.

Preparation removes duplicate rows and rows with missing prices, then keeps apartment listings with monthly prices and no fee. This produces 99,141 observations.

### Model features

- Bathrooms, bedrooms, and floor area (`square_feet`).
- Latitude and longitude.
- Sine and cosine of the listing month.
- Photo availability (`has_photo`).
- Parking, recreation, and comfort indicators extracted from amenities.

Missing amenities produce zero-valued indicators: these indicate that the corresponding amenity was not detected in the recorded text.

## Methodology

1. Prepare the analytics base table (ABT).
2. Split the data into approximately **70% training, 15% validation, and 15% test**, using `random_state=42`.
3. Fill missing numerical features with medians calculated from the training set.
4. Prepare model-specific features:
   - **KNN:** one-hot encode `has_photo` and standardize seven numerical features using a scaler fitted on training data.
   - **Custom Decision Tree:** retain numerical values and categorical `has_photo` values.
   - **scikit-learn Decision Tree and Random Forest:** one-hot encode `has_photo` and retain the original numerical scale.
5. Compare hyperparameters using validation MAE and MAPE.
6. Evaluate the selected configurations on the test set.

Final models are fitted on the training set only. The test set is reserved for final evaluation and must not be used to choose hyperparameters.

## Models and Experiments

The custom KNN calculates Euclidean distances directly and predicts the mean price of the selected neighbours. The custom Decision Tree uses numerical threshold splits, categorical multi-way splits, weighted MSE as its split criterion, and mean-valued leaves.

| Model | Validation experiments | Selected parameters |
|---|---|---|
| Custom KNN | `k=1…15` | `k=2` |
| scikit-learn KNN | `k=1…15` | `k=2` |
| Custom Decision Tree | Depth: 25, 30, 35; split: 5, 10; leaf: 2, 5 | Depth 35, split 5, leaf 2 |
| scikit-learn Decision Tree | Depth: 25, 30, 35; split: 5, 10; leaf: 2, 5 | Depth 30, split 5, leaf 2 |
| Random Forest | Trees: 50, 100, 200, 300; depth: 10, 20, 30, 40, None | 300 trees, unlimited depth |

The selected parameters are explicitly set in `src/final_evaluation.py`; the pipeline does not automatically select and transfer the best configuration from the tuning tables.

## Setup

Use Python 3.12, as used for the current project environment. Run all commands from the project root (`Lab_1`), because file paths are relative to that directory.

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the dataset is still compressed, extract the supplied 100K archive into `data/` so that the CSV path above exists.

## Running the Project

### 1. Explore the data

```bash
python run_analysis.py
```

This prints dataset information, missing values, descriptive statistics, extreme listings, split sizes, and preprocessing checks. It saves histograms, boxplots, scatterplots, and categorical bar charts under `outputs/plots/raw_data/`.

Existing quality-summary CSVs are also available under `outputs/tables/` and `outputs/plots/raw_data/tables/`. The current analysis script prints its quality tables; it does not export those CSVs automatically.

### 2. Run validation experiments and final evaluation

```bash
mkdir -p outputs/custom_knn
python main.py
```

The full pipeline executes all validation experiments, prints their results, saves tuning CSVs, and then evaluates all five selected models on the test set.

**This command includes expensive Custom Decision Tree and Random Forest experiments.** Existing neighbour caches avoid repeated Custom KNN neighbour searches, but do not skip tuning or tree/forest training. Each tuning table is saved after its experiment finishes; the final table is saved after final evaluation finishes.

### 3. Generate tuning plots from saved results

```bash
python src/report_outputs.py
```

This script reads the tuning CSVs and generates three MAE plots. It does not train models or rerun experiments. Missing input files are reported in the terminal, and the corresponding plot is skipped.

For a machine without a graphical display:

```bash
MPLBACKEND=Agg python src/report_outputs.py
```

## Saved Outputs

### Hyperparameter tables

The simple `save_results()` function converts each existing list of result dictionaries to a pandas DataFrame and saves it without an index.

```text
outputs/tuning/
├── custom_knn_tuning.csv
├── sklearn_knn_tuning.csv
├── custom_dt_tuning.csv
├── sklearn_dt_tuning.csv
└── random_forest_tuning.csv
```

KNN tables contain `mae` and `mape`; tree and forest tables contain `MAE` and `MAPE`. Custom KNN uses the column `k`, while scikit-learn KNN uses `k value`. The plotting script handles these existing column names.

### Tuning plots

```text
outputs/tuning/plots/
├── knn_mae_vs_k.png
├── decision_tree_mae_vs_depth.png
└── random_forest_mae.png
```

- **KNN:** compares both implementations across `k`, with `k=2` marked.
- **Decision Tree:** compares depths for `min_samples_split=5` and `min_samples_leaf=2`, with selected depths marked.
- **Random Forest:** compares estimator counts for each tested depth, with 300 trees and unlimited depth marked.

### Final test results

The final table is saved to `outputs/final/test_results.csv`.

| Model | MAE (USD) | MAPE (%) |
|---|---:|---:|
| Custom KNN | 291.399408 | 19.151561 |
| scikit-learn KNN | 291.290647 | 19.146568 |
| Custom Decision Tree | 220.786675 | 14.093724 |
| scikit-learn Decision Tree | 221.636363 | 14.119564 |
| Random Forest | 170.018859 | 11.102524 |

MAE is the mean absolute price error. MAPE is the mean absolute percentage error, multiplied by 100. MSE is used internally to choose Custom Decision Tree splits; it is not an exported evaluation metric.

The `Time` column is not a consistent runtime benchmark: Custom KNN excludes neighbour search, scikit-learn KNN has no recorded time, and tree/forest measurements include fitting and prediction.

Running the scripts again overwrites their corresponding CSV and PNG outputs.

## Custom KNN Neighbour Cache

```text
outputs/custom_knn/
├── val_neighbors_k50.npy
└── test_neighbors_k2.npy
```

These files store indices of training observations nearest to each validation or test observation. They do not store predictions or evaluation metrics.

If a cache exists, the pipeline loads it with `np.load()`. Otherwise, it calculates neighbours and saves them with `np.save()`.

If the dataset, row order, split, preprocessing, or required neighbour count changes, remove the affected cache before rerunning. Cache validity is not checked automatically. No cache deletion is needed just to save tuning tables or regenerate report plots.

## Project Structure

```text
Lab_1/
├── data/                      # Dataset CSV and archives
├── outputs/
│   ├── custom_knn/            # Neighbour caches
│   ├── final/                 # Final test results
│   ├── plots/raw_data/        # Exploratory plots and earlier tables
│   ├── tables/                # Existing data-quality tables
│   └── tuning/                # Validation experiment tables and plots
├── src/
│   ├── preprocessing.py       # ABT, splitting, imputation, encoding, scaling
│   ├── data_analysis.py       # Quality summaries and exploratory plots
│   ├── custom_knn.py          # KNN implementation from scratch
│   ├── knn_tunning.py         # Custom KNN validation experiments
│   ├── sklearn_knn.py         # scikit-learn KNN evaluation
│   ├── custom_decision_tree.py # Decision Tree implementation from scratch
│   ├── dt_tunning.py          # Custom Decision Tree validation experiments
│   ├── sklearn_dt.py          # scikit-learn DT evaluation and experiments
│   ├── random_forest.py       # Random Forest evaluation and experiments
│   ├── evaluation.py          # Shared MAE and MAPE calculation
│   ├── final_evaluation.py    # Fixed-parameter test evaluation
│   ├── save_results.py        # Simple tuning CSV export
│   └── report_outputs.py      # Plots from saved tuning CSVs
├── main.py                    # Full experiment pipeline
├── run_analysis.py            # Data analysis and preprocessing inspection
├── requirements.txt           # Pinned dependencies
└── README.md

```

