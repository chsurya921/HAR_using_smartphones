# HAR with Hidden Markov Models and Ensemble Learning

**Intelligent Human Activity Recognition Using Hidden Markov Models and Ensemble Learning: A Comparative and Hybrid Modelling Study**

B.Tech / M.Tech (Integrated) CSE, Regulations 2021 — SRMIST

---

## Project Overview

This project implements a complete, reproducible pipeline for Human Activity Recognition (HAR) using the UCI HAR smartphone dataset. It combines:
- **Unit 4:** Hidden Markov Models built from scratch in NumPy (forward/backward/Viterbi/Baum-Welch)
- **Unit 5:** Ensemble methods (Random Forest, Gradient Boosting, Voting, Stacking)
- **Hybrid:** HMM-Ensemble hybrid using ensemble posteriors as HMM emission evidence
- **Website:** Streamlit interactive demo
- **Report:** Fully-populated with measured experimental results

---

## Quick Setup

```bash
# 1. Clone or place in your working directory
cd HAR_HMM_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all tests
cd tests && pytest -v

# 4. Run full experiment pipeline
python src/experiment_runner.py

# 5. Launch the website demo
streamlit run streamlit_app/app.py
```

---

## Project Structure

```
HAR_HMM_Project/
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.yaml      # All seeds, paths, hyperparameter ranges (frozen)
├── src/
│   ├── data_loader.py              # Subject-independent splits, sequence builder
│   ├── preprocessing.py            # StandardScaler + PCA (fit on train only)
│   ├── hmm_algorithms.py           # Scaled forward, backward, Viterbi (NumPy)
│   ├── hmm_model.py                # Full HMM with Baum-Welch, Gaussian/GMM emissions
│   ├── ensemble_models.py          # LR, DT, RF, HistGBT, Voting, Stacking
│   ├── hybrid_model.py             # HMM-Ensemble hybrid (emission evidence method)
│   ├── evaluation.py               # All metrics, confusion matrices
│   ├── visualization.py            # All figures (generated from results/ only)
│   └── experiment_runner.py        # Experiments A–I
├── tests/
│   ├── test_hmm_algorithms.py      # Unit tests for HMM (normalisation, Viterbi, etc.)
│   ├── test_preprocessing.py       # Leakage tests, split validity
│   └── test_model_outputs.py       # Probability normalisation, output shape tests
├── results/
│   ├── metrics.csv                 # All experiment results
│   ├── figures/                    # All saved plots
│   └── logs/                       # Experiment logs
├── report/
│   └── report.md                   # Full technical report (filled from results/)
└── streamlit_app/
    └── app.py                      # Interactive website demo
```

---

## Dataset

UCI Human Activity Recognition Using Smartphones  
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

- **561** pre-engineered features per 2.56-second window (50% overlap)
- **30 subjects**, **6 activities**: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- **Train:** 7,352 samples (21 subjects) | **Test:** 2,947 samples (9 subjects)

---

## Reproducibility

All seeds are fixed in `config/experiment_config.yaml`. Given the same Python/package versions and that config, results are reproducible to numerical precision.
