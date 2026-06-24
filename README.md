# NBA Intelligent Scheduling & Prediction System

Project for the **Knowledge Engineering** course — University of Bari Aldo Moro.

The system integrates three AI/KBS components applied to the NBA domain:
**supervised learning** for outcome prediction,
**constrained optimization** for TV scheduling,
**informed search** for logistics planning.

---

## Project Structure

```text
NBA-Intelligent-Scheduling-Prediction-System/
│
├── 1_data_preparation/             # Data ingestion, assessment, cleaning
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_assessment.ipynb
│   └── 03_data_cleaning.ipynb
│
├── 2_data_analysis/                # EDA and feature engineering
│   ├── 01_eda_univariate.ipynb
│   ├── 02_eda_bivariate.ipynb
│   └── 03_feature_engineering.ipynb
│
├── notebooks-ML/                   # Supervised learning model training
│   └── model_training_comparison.ipynb
│
├── notebooks-CSP/
│   └── nba_scheduling_model_guide.ipynb          # TV scheduling CSP/COP with real NBA data
│
├── notebooks-optimization/
│   └── road_trip_astar_optimizer.ipynb           # A* road trip optimizer
│
├── src/                            # Reusable Python modules
│   ├── ml/                         # data_loader, preprocessing, trainer, evaluator
│   └── loader.py
│
├── data/
│   ├── 1_raw/                      # Season-level raw NBA data
│   ├── 2_processed/                # Cleaned dataset
│   └── 3_features/                 # Feature-engineered dataset shared by ML and CSP
│
├── config.toml                     # Centralized configuration
└── requirements.txt

```

---

## System Components

### 1. ML Pipeline — NBA Outcome Prediction

**Notebook**: `notebooks-ML/model_training_comparison.ipynb`

**Dataset**: 31,160 NBA games (2000-01 → 2025-26 seasons)

**Task**: classification (`home_win`) and regression (`point_differential`)

**Temporal split** (no data leakage):

| Split | Seasons | Games |
| --- | --- | --- |
| Train | 2000-01 → 2018-19 | 22,881 |
| Val | 2019-20 → 2021-22 | 3,369 |
| Test | 2022-23 → 2025-26 | 4,910 |

**Models compared**: Random Forest, XGBoost, Gradient Boosting

**Features**: 42 rolling features (net rating, win rate, pace, TS%, back-to-back, streak, rest days, ...)

**Evaluation**: walk-forward cross-validation (5 temporal folds) → metrics such as μ ± σ

### 2. CSP/COP — NBA TV Scheduling

**Notebook**: `notebooks-CSP/nba_scheduling_model_guide.ipynb`

**Technique**: CP-SAT (Google OR-Tools) — CDCL + constraint propagation

**Problem**: assigning $|M|$ games to $|S|$ TV slots while respecting operational and business constraints

**Dataset link**: the CSP instance now uses 13 real NBA games from the 2022-23 season, loaded from the same feature-engineered dataset used by the ML pipeline (`features_nba_data_2000-01_2025-26.csv`). The `is_big_match` label is derived from `home_winrate_last_10` and `away_winrate_last_10`, making the scheduling model consistent with the project's shared data layer.

**Knowledge Base**:

* **Variables**: $|M| \times |S|$ binary BoolVars ($x_{m,s} \in \{0,1\}$)
* **Hard constraints**: assignment (each game in exactly 1 slot), slot capacity (≤ 1 game/slot), team conflict (no team plays twice on the same day)
* **Business constraints**: prime-time capacity cap, no-overlap for big matches on the same channel
* **Objective**: maximize $\sum_{m \in M^B} \sum_{s \in S^P} x_{m,s}$ (big matches in prime time)

The problem belongs to the NP-hard class (scheduling with resource constraints). CP-SAT solves it via propagation + branch-and-bound on instances of this size almost instantly.

### 3. A* Road Trip Optimizer

**Notebook**: `notebooks-optimization/road_trip_astar_optimizer.ipynb`

**Problem**: TSP on 18 nodes (NBA arenas) — finding the minimum cost road trip

**Costs**: Haversine distance × 1.5 + fixed logistics costs per destination

**Algorithm**:

* State: `(current_node, unvisited_cities_tuple)`
* State space: $n \cdot 2^{n-1}$ = 2,359,296 theoretical states for $n=18$
* **Heuristic**: `h(n) = MST(R) + min_dist(current, R) + min_dist(Home, R) + Σ fixed_costs(R)`
* Admissibility and consistency formally proven
* Memoization with `lru_cache` for MST and heuristic calls
* Initial upper bound: nearest neighbor + 2-opt (branch & bound)

> **Note on Map Visualization**: If the interactive map (generated with `folium`) does not display correctly when opening the notebook, Jupyter might be blocking HTML/JavaScript elements for security reasons. To resolve this, you need to mark the notebook as "Trusted" by running the following command in your terminal:
>
> ```bash
> jupyter trust notebooks-optimization/road_trip_astar_optimizer.ipynb
> ```

---

## Interconnection between components

The three modules operate on complementary aspects of the NBA domain:

```text
[ML Dataset]  Provides real NBA games and rolling team-form features
  └─→  [CSP]  Builds a feasible TV schedule for selected real games
         └─→  [ML]  Could evaluate scheduled games with predictive features
                       (home_winrate_last_10, away_winrate_last_10, B2B indicators)

[A*]   Optimizes team road trips
  └─→  road trips influence logistics costs
         └─→  [CSP]  could use A* distances as weights in constraints

```

The shared feature dataset now connects the ML and CSP modules directly: ML notebooks
create rolling team-form features, while the CSP notebook reuses those features to
identify high-profile games in a real 2022-23 scheduling sample. In a fully integrated
system, the CSP output could feed ML prediction and the A* output would inform
scheduling costs, for example by limiting back-to-backs caused by long road trips.

---

## Dataset

| File | Rows | Description |
| --- | --- | --- |
| `data/1_raw/seasons/nba_season_*.csv` | ~63,000 | Season-level raw data collected with `nba_api` |
| `data/2_processed/cleaned_nba_data_2000-01_2025-26.csv` | ~31,000 | Cleaned and validated data |
| `data/3_features/features_nba_data_2000-01_2025-26.csv` | 31,160 | Feature engineered + label split; also used by the CSP notebook to load real 2022-23 games and derive `is_big_match` |

Data source: `nba_api` (BoxScoreAdvancedV3, BoxScoreTraditionalV3)

---

## Installation

```bash
pip install -r requirements.txt

```

Main requirements: `polars`, `pandas`, `scikit-learn`, `xgboost`, `ortools`, `nba_api`, `networkx`, `folium`, `plotly`

---

## Knowledge Engineering Topics Covered

* **Supervised learning**: classification and regression with ensemble methods, evaluation with temporal walk-forward CV
* **Informed search**: A* with admissible and consistent heuristic (MST lower bound) on an exponential state space
* **Constraint Satisfaction / Optimization**: binary CSP modeling, constraint propagation, combinatorial optimization with CP-SAT
