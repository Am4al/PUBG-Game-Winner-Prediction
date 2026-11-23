# 🎮 PRCP-1012: PUBG Game Winner Prediction

An end-to-end Machine Learning pipeline and predictive analytics project to predict the win placement percentile (`winPlacePerc`) of players in a PUBG match and identify the critical gameplay factors that drive match victory.

---

## 📌 Project Overview
* **Project Code:** PRCP-1012-GameWinnerPred
* **Task Type:** Continuous Regression
* **Target Variable:** `winPlacePerc` (continuous percentile score between `0.0` [last place] and `1.0` [1st place/winner])
* **Primary Objective:** Build a predictive model to accurately estimate match finish placement and analyze the strategic in-game factors that maximize the probability of winning.

---

## 📊 Dataset Overview
The dataset contains **4,446,966 records** and **29 columns**, representing player statistics across diverse PUBG game modes (Solo, Duo, Squad, FPP/TPP).

### Key Features:
* **Mobility:** `walkDistance`, `rideDistance`, `swimDistance`
* **Combat:** `kills`, `damageDealt`, `DBNOs` (knockdowns), `headshotKills`, `killPlace`, `killStreaks`, `longestKill`
* **Survival & Support:** `heals`, `boosts`, `revives`, `weaponsAcquired`
* **Match Metadata:** `matchId`, `groupId`, `matchType`, `matchDuration`, `numGroups`, `maxPlace`
* **Target:** `winPlacePerc` (placement percentile)

---

## 🛠️ Project Architecture & Methodology

```
Raw Data (4.45M Rows)
       │
       ▼
[1. Memory Downcasting] ➔ 65% RAM reduction (float64 ➔ float32, int64 ➔ int16/int8)
       │
       ▼
[2. Exploratory Data Analysis (Task 1)] ➔ 8 visual studies on mobility, combat, item usage, & correlations
       │
       ▼
[3. Data Cleaning] ➔ Removed 1 missing target row; verified finite values
       │
       ▼
[4. Feature Engineering] ➔ Group-level aggregations (mean, sum), totalDistance, itemsUsed, killDamageRatio
       │
       ▼
[5. Match-Aware Splitting] ➔ GroupShuffleSplit on matchId (80% Train / 20% Test, ZERO match leakage)
       │
       ▼
[6. Multi-Model Benchmarking (Task 2)] ➔ Ridge, Decision Tree, Random Forest, LightGBM, XGBoost
       │
       ▼
[7. Feature Importance Analysis] ➔ Identified Top 10 factors affecting win probability
       │
       ▼
[8. Production Recommendation & Challenges Reports]
```

---

## 🏆 Model Benchmarking & Performance Comparison

Evaluated on a strictly isolated **50,000-sample holdout test set** with zero match overlap:

| Rank | Model | Category | MAE (↓) | RMSE (↓) | $R^2$ Score (↑) | Training Time (s) | Inference Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost** | Gradient Boosting | **0.04921** | **0.06807** | **0.9513** | 39.12 s | 0.0273 ms/sample |
| 🥈 | **LightGBM** | Gradient Boosting | **0.04952** | **0.06819** | **0.9511** | **30.90 s** | 0.0584 ms/sample |
| 🥉 | **Random Forest** | Bagging Ensemble | 0.05590 | 0.07941 | 0.9337 | 416.32 s | 0.0107 ms/sample |
| 4 | **Decision Tree** | Single Tree | 0.05960 | 0.08463 | 0.9247 | 10.87 s | 0.0003 ms/sample |
| 5 | **Ridge Regression** | Linear Baseline | 0.07083 | 0.09695 | 0.9012 | **1.76 s** | **0.0002 ms/sample** |

### 🚀 Production Model Recommendation: **LightGBM Regressor**
* **Predictive Accuracy:** Achieves near-identical accuracy to XGBoost ($R^2 = 0.9511$, $\text{MAE} = 0.04952$), deviating by less than **4.95%** from actual placement percentiles.
* **Retraining Speed:** Trains **~21% faster** than XGBoost and **13× faster** than Random Forest.
* **Low Memory Footprint:** Histogram-based binning requires minimal RAM overhead, making it ideal for microservice deployment and high-throughput real-time game serving.

---

## 🎯 Key Factors Driving Win Placement (Task 2 Deliverable)

Feature importance analysis reveals that winning in PUBG is overwhelmingly determined by **team movement, survival positioning, and health maintenance**, rather than pure aggressive combat:

| Rank | Feature | Importance Share | Strategic Insight |
| :---: | :--- | :---: | :--- |
| **1** | `group_walkDistance_mean` | **50.74%** | Squad movement and play zone rotation is the single biggest driver of victory. |
| **2** | `group_totalDistance_mean` | **20.92%** | Total squad relocation distance ensures late-game survival in closing circles. |
| **3** | `killPlace` | **6.86%** | Match kill ranking outranks raw kill count—timing of engagements matters most. |
| **4** | `group_boosts_mean` | **4.16%** | Health regeneration and speed boosts keep squads alive through blue zone damage. |
| **5** | `walkDistance` | **3.10%** | Individual player ground mobility. |
| **6** | `kills` | **1.32%** | Individual frag count (secondary to movement and survival). |

> **Key Takeaway:** Movement and squad positioning account for **over 74%** of the model's total predictive power.

---

## 💡 Challenges Faced & Solutions

1. **Massive Dataset Scale (4.45M rows):**
   * *Solution:* Downcasted data types from 64-bit to 8/16/32-bit (saving 65%+ RAM) and used calibrated match-aware subsampling for rapid multi-model iteration.
2. **Match-Level Data Leakage:**
   * *Solution:* Used `GroupShuffleSplit` on `matchId` to guarantee that all players from a given match remain quarantined in either train or test, never both.
3. **Squad/Team Interdependence:**
   * *Solution:* Engineered group-level aggregates (`mean`, `sum`) for mobility, combat, and healing stats, reflecting team strength.
4. **Prediction Bounds Compliance:**
   * *Solution:* Applied `np.clip(y_pred, 0.0, 1.0)` post-processing to enforce physical probability bounds.

---

## 📁 Repository Structure

```
├── Final_PUBG_GameWinnerPred.ipynb    # Complete executed Jupyter Notebook (all tasks & outputs)
├── PRCP-1012-GameWinnerPred.docx      # Official problem statement & project specification
├── README.md                          # Comprehensive project documentation
└── .gitignore                         # Excludes large data files (>100 MB)
```

---

## 💻 How to Run

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Final_PUBG_GameWinnerPred.ipynb`.
3. Set runtime to **CPU** (**Runtime → Change runtime type → None**).
4. Run all cells (**Runtime → Run All**). The notebook will auto-download the dataset if not found locally.

### Option 2: Local Jupyter Notebook
```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost

# Launch Jupyter
jupyter notebook Final_PUBG_GameWinnerPred.ipynb
```

---

