# MarketingMLProject

An end-to-end Machine Learning pipeline utilizing US Census data to deliver a population-weighted classification model and an unsupervised consumer segmentation model for strategic target marketing.

## Project Objectives
* Predictive Income Classification Model: An optimized gradient boosted framework (`XGBoost`) capable of predicting whether a citizen's annual income sits below or above $50K, factoring in 40 demographic, sociaeconomic, and employment features.
* Customer Profile Segmentation Model: A distance-based pipeline (`K-Means`) used to cluster real-world weighted citizens into 5 actionable customer personas

---

## Repository Structure
```text
MarketingMlProject/
│
├── data/
│   ├── census-bureau.columns      # Column descriptive headers metadata
│   └── census-bureau.data         # Raw dataset file
│
├── outputs/
│   ├── feature_target_correlation.png          # Target Pearson correlation bars
│   ├── global_correlation_matrix.png           # Multicollinearity global heatmap
│   └── final_weighted_confusion_matrix.png     # Population-scaled holdout matrix
│   └── kmeans_elbow_plot.png                   # WCSS distortion curve
│   └── inertia_drop_plot.png                   # Delta information gain bar chart
│
├── src/
│   ├── init.py
│   ├── utils.py                   # Data-loading and cleaning utilities
│   ├── train_classifier.py        # Objective 1: Classification Task
│   └── run_segmentation.py        # Objective 2: Clustering/Segmentation Task
│
├── DataObservationAndPlanning.ipynb        # Sandbox notebook containing intial data observation and planning
├── environment.yml                         # Cross-platform Conda environment profile
├── README.md                               # Deployment & execution instructions
├── report.pdf                              # Detailed project report
└── requirements.txt                        # Dependencies file
```

---

## Environment Setup and Installation
This project requires Python 3.10+ and utilizes an environment framework to secure underlying package distributions.

**Option A: Restore via Conda (Recommended for Anaconda users)**
To rebuild the cross-platform environment exactly as it was compiled during planning, use the `environment.yml` profile:

```bash
conda env create -f environment.yml
conda activate base
```

**Option B: Restore via Pip Standard Environment**
If you are deploying outside of a Conda landscape, create your own virtual environment and target the requirements block:
```bash
git clone [https://github.com/dorybphp/MarketingMLProject.git](https://github.com/dorybphp/MarketingMLProject.git)
cd MarketingMLProject
pip install -r requirements.txt
```

---

## Execution Instructions
Run these commands from the root of your project directory within your activated environment terminal.

### Run Income Classification (Supervised Learning)
To clean categorical records, generate interaction terms, computer Pearson correlation arrays, trigger the 3-fold stratified cross-validarion parameter grid sweep, and evaluate metrics:
```bash
python src/classifier.py
```
* Expected Console Readouts: Displays Grid Search status loops, the optimized hyperparameter map configurations, feature importance rankings, and the final holdout population-weighted `classification_report`
* Saved Outputs: Check your `outputs/` folder for `feature_target_correlation.png`, `global_correlation_matrix.png`, and `final_weighted_confusion_matrix.png`

### Run Customer Segmentation (Unsupervised Clustering)
To apply median/constant imputation, transform categorical dimensions via OHE spatial vectors, render inertia charts, and fit the final population-weighted $K = 5$ model:
```bash
python src/segmentation.py
```
* Expected Console Readouts: Renderss numerical centroid averages, calculated true-population scale volumes per segment, and the dominant text drivers per persona.
* Saved Outputs: Check your `outputs/` folder for `kmeans_elbow_plot.png` and `inertia_drop_plot.png`

---

## Consumer Personas Index (Summary Profiles)
| Cluster ID | Core Demographic | Workforce Engagement | Key Financial Driver |
| :--------- | :--------------- | :------------------- | :------------------- |
| 0          | Peak Career (30-54) | Private Sector/Clerical & Admin | Consistent Hourly Wages |
| 1          | Youth (0-17) | Not in Workforce | Household Dependent Spend |
| 2          | Peak Career (30-54) | Professional Specialties / Execs | Large Capital Gains & Dividends |
| 3          | Seniors (55+) | Retired/Outside Labor Force | Stable Asset Dividends |
| 4          | Young Adult/Peak Career | Private Sector/Clerical & Admin | Full-Time Earned Salaries |

---

## Built With
* **XGBoost**: Gradient boosted tree frameworks optimizing sample-weighted log-loss arrays.
* **Scikit-Learn**: Core pipeline assembly, data transformations (`StandardScaler`, `OneHotEncoder`), and predictive metrics evaluations.
* **Pandas & NumPy**: In-memory population matrix scaling, aggregations, and transpositions.
* **Matplotlib & Seaborn**: Vectorized visualization layouts for data audits.