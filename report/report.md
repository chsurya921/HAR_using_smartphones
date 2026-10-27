# Project Report: Intelligent Human Activity Recognition Using Hidden Markov Models and Ensemble Learning: A Comparative and Hybrid Modelling Study

**Degree & Regulations:** B.Tech / M.Tech (Integrated) CSE, Regulations 2021  
**Institution:** SRM Institute of Science and Technology  
**Authors:** Student Project Team  
**Date:** October 2026  
**Computing Environment:** Apple Silicon ARM64 (CPU Only, zero GPU dependence), Python 3.13.15  

---

## Abstract

Human Activity Recognition (HAR) using body-worn inertial sensors is a cornerstone of ubiquitous computing, assistive healthcare, and sports biomechanics. Conventional machine learning classifiers classify sensor windows in isolation, discarding the continuous temporal dynamics inherent in human physical movement. Conversely, generative sequential models such as Hidden Markov Models (HMMs) model temporal transitions explicitly but struggle with high-dimensional feature spaces due to restrictive parametric emission assumptions. 

This study presents a comparative and hybrid modelling methodology on the benchmark UCI HAR dataset (10,299 windowed samples, 30 subjects, 6 activity classes). We build and evaluate:
1. Conventional discriminative baselines (Multinomial Logistic Regression, CART Decision Trees);
2. Ensemble architectures spanning Bagging (Random Forest), Gradient Boosting (Histogram Gradient Boosting), Hard/Soft Voting, and Out-of-Fold Stacking;
3. A from-scratch, numerically scaled Hidden Markov Model (Unit 4: Forward-Backward, Scaling Factors, Viterbi, and Baum-Welch EM) implemented in pure NumPy;
4. A continuous Linear Dynamical System (LDS) Kalman Filter demo on raw inertial signals;
5. An **HMM-Ensemble Hybrid** that unifies discriminative class posteriors as emission evidence within a temporal Markov transition lattice ($P(y_t \mid x_t) / P(y_t)$).

All models were evaluated under strict, zero-leakage, subject-independent partitions (Train: 17 subjects, Validation: 4 subjects, Test: 9 subjects). Empirical results demonstrate that while the standalone Gaussian HMM achieves **68.42%** test macro-F1 and the top-performing stacking ensemble achieves **86.40%** macro-F1, the **Hybrid HMM-Ensemble achieves 93.44% test macro-F1 (93.62% test accuracy)**. Furthermore, sequential evaluation reveals a **14.4× reduction in false activity switch rate** (from 0.0933 down to 0.0065) and a **transition consistency of 80.18%**, resolving spurious frame flickers and providing physically plausible movement timelines.

---

## 1. Introduction and Theoretical Foundations

### 1.1 Problem Formulation
Let $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T]^\top \in \mathbb{R}^{T \times D}$ denote a temporal sequence of feature vectors extracted from sliding windows of triaxial accelerometer and gyroscope data. Each window corresponds to an underlying true human activity $y_t \in \{1, 2, \dots, K\}$ where $K=6$:
1. `WALKING`
2. `WALKING_UPSTAIRS`
3. `WALKING_DOWNSTAIRS`
4. `SITTING`
5. `STANDING`
6. `LAYING`

Conventional supervised learning models predict each label $\hat{y}_t = f(\mathbf{x}_t)$ independently, assuming independent and identically distributed (i.i.d.) observations:
$$P(y_1, \dots, y_T \mid \mathbf{x}_1, \dots, \mathbf{x}_T) = \prod_{t=1}^T P(y_t \mid \mathbf{x}_t)$$
This assumption violates human kinematics: a person cannot realistically alternate between `LAYING` and `WALKING_UPSTAIRS` within a single 2.56-second window without passing through intermediate postural transitions. 

### 1.2 Syllabus-to-Implementation Mapping

| Syllabus Topic | Theoretical & Mathematical Framework | Project Implementation |
| :--- | :--- | :--- |
| **HMM Definition** (Unit 4) | Triple $\lambda = (\boldsymbol{\pi}, \mathbf{A}, \mathbf{B})$; first-order Markov property $P(q_t \mid q_{1:t-1}) = P(q_t \mid q_{t-1})$; conditional independence of observations | `src/hmm_algorithms.py`: Pure NumPy vectorization, row-stochastic validation |
| **Likelihood & Scaling** (Unit 4) | Forward variable $\alpha_t(i)$, backward variable $\beta_t(i)$; scaling coefficients $c_t = \sum_i \alpha_t(i)$ to prevent IEEE 754 underflow | `scaled_forward()`, `scaled_backward()`, `log_likelihood()`; verified against log-space forward |
| **Optimal Path Decoding** (Unit 4) | Dynamic programming trellis $\delta_t(j) = \max_i [\delta_{t-1}(i) + \ln A_{ij}] + \ln B_j(\mathbf{x}_t)$; backpointer matrix $\psi_t(j)$ | `viterbi()` log-space implementation with backward traceback |
| **Parameter Estimation** (Unit 4) | Supervised Maximum Likelihood Estimation (MLE) and unsupervised Baum-Welch (EM) expectation-maximization | `_init_supervised()` and multi-sequence `fit()` Baum-Welch in `src/hmm_model.py` |
| **Continuous Latent Space** (Unit 4) | Linear Dynamical Systems (LDS): Continuous state Gaussian transition and measurement models | `src/kalman_filter.py`: 1D Kalman filter tracking latent body acceleration on raw inertial signals |
| **Dimensionality Reduction** (Unit 3) | Principal Component Analysis (PCA) preserving orthogonal maximal variance directions | `src/preprocessing.py`: StandardScaler + 20-component PCA fit on training partition only |
| **Ensemble Learning** (Unit 5) | Bagging (bootstrap aggregation), Boosting (sequential gradient boosting), Voting (hard/soft), Stacking (meta-learning) | `src/ensemble_models.py`: LR, DT, RF, HistGBT, Voting, and Stacking |
| **Hybrid System** | Emission Evidence Substitution: scaled likelihood $L(\mathbf{x}_t \mid s) \propto P(y=s \mid \mathbf{x}_t) / P(y=s)$ | `src/hybrid_model.py`: Soft ensemble posteriors fed into Viterbi decoder with Markov transition matrix |

---

## 2. Mathematical Formulations (Unit 4 & Hybrid)

### 2.1 Hidden Markov Model (HMM)
A discrete-state Hidden Markov Model is specified by:
1. **Initial state distribution:** $\boldsymbol{\pi} \in \mathbb{R}^K$, where $\pi_i = P(q_1 = i)$ with $\sum_{i=1}^K \pi_i = 1$.
2. **State transition probability matrix:** $\mathbf{A} \in \mathbb{R}^{K \times K}$, where $A_{ij} = P(q_{t+1} = j \mid q_t = i)$ with $\sum_{j=1}^K A_{ij} = 1$.
3. **Observation emission density:** $B_j(\mathbf{x}_t) = p(\mathbf{x}_t \mid q_t = j)$. In this study, we utilize Gaussian emissions:
   $$\mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}_j|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}_t - \boldsymbol{\mu}_j)^\top \boldsymbol{\Sigma}_j^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_j)\right)$$
   To avoid singularity in high-dimensional feature spaces, a variance floor $\epsilon = 10^{-5}$ is enforced along the diagonal:
   $$\boldsymbol{\Sigma}_{j, dd} \leftarrow \max(\boldsymbol{\Sigma}_{j, dd}, \epsilon)$$

### 2.2 Scaled Forward-Backward Algorithm
To prevent severe numerical underflow over sequences of length $T > 100$, scaling factors $c_t$ normalize the forward variables at every time step:
$$\alpha_1(i) = \pi_i B_i(\mathbf{x}_1), \quad c_1 = \sum_{i=1}^K \alpha_1(i), \quad \tilde{\alpha}_1(i) = \frac{\alpha_1(i)}{c_1}$$
$$\alpha_t(j) = \left[\sum_{i=1}^K \tilde{\alpha}_{t-1}(i) A_{ij}\right] B_j(\mathbf{x}_t), \quad c_t = \sum_{j=1}^K \alpha_t(j), \quad \tilde{\alpha}_t(j) = \frac{\alpha_t(j)}{c_t}$$
The total sequence log-likelihood is directly obtained from the scaling factors:
$$\ln P(\mathbf{X} \mid \lambda) = \sum_{t=1}^T \ln c_t$$
The backward algorithm uses the exact same scaling factors $c_t$:
$$\tilde{\beta}_T(i) = 1 / c_T$$
$$\tilde{\beta}_t(i) = \sum_{j=1}^K A_{ij} B_j(\mathbf{x}_{t+1}) \tilde{\beta}_{t+1}(j) / c_t$$
Posterior state probabilities $\gamma_t(i) = P(q_t = i \mid \mathbf{X}, \lambda)$ are obtained via the smoothed product:
$$\gamma_t(i) = \frac{\tilde{\alpha}_t(i) \tilde{\beta}_t(i)}{\sum_{j=1}^K \tilde{\alpha}_t(j) \tilde{\beta}_t(j)}$$

### 2.3 Viterbi Decoding
The maximum a posteriori (MAP) state sequence is computed via dynamic programming in the log domain:
$$\delta_1(i) = \ln \pi_i + \ln B_i(\mathbf{x}_1)$$
$$\delta_t(j) = \max_{1 \le i \le K} \left[\delta_{t-1}(i) + \ln A_{ij}\right] + \ln B_j(\mathbf{x}_t)$$
$$\psi_t(j) = \arg\max_{1 \le i \le K} \left[\delta_{t-1}(i) + \ln A_{ij}\right]$$
Traceback from $t=T$ to $1$:
$$q^*_T = \arg\max_{1 \le i \le K} \delta_T(i), \quad q^*_t = \psi_{t+1}(q^*_{t+1})$$

### 2.4 Hybrid HMM-Ensemble (Emission Evidence Substitution)
The core limitation of the classical Gaussian HMM on 561-dimensional HAR features is that unimodal or mixture Gaussian emissions fail to capture complex, non-linear decision boundaries. Conversely, ensemble classifiers (Random Forests, Gradient Boosting) excel at partition-based boundary estimation but lack temporal transition priors.

To unify both paradigms, we employ Bayes' rule to invert the ensemble's posterior probability:
$$P(y_t = s \mid \mathbf{x}_t) = \frac{p(\mathbf{x}_t \mid y_t = s) P(y_t = s)}{p(\mathbf{x}_t)}$$
Rearranging for the scaled likelihood:
$$L(\mathbf{x}_t \mid y_t = s) = \frac{p(\mathbf{x}_t \mid y_t = s)}{p(\mathbf{x}_t)} = \frac{P(y_t = s \mid \mathbf{x}_t)}{P(y_t = s)}$$
Where $P(y_t = s \mid \mathbf{x}_t)$ is estimated by the soft-voting ensemble (`SoftVotingClassifier`) and $P(y_t = s)$ is the empirical class prior computed on training data. These scaled likelihoods substitute for $B_s(\mathbf{x}_t)$ within the Viterbi trellis, evaluated alongside the sequence transition matrix $\mathbf{A}$.

---

## 3. Experimental Protocol & Leakage Protection

### 3.1 Strict Subject-Independent Partitions
In wearable sensor computing, partitioning windows randomly across train and test causes catastrophic feature leakage due to identical participant anthropometrics, walking cadence, and sensor orientation. 

To ensure complete leakage safety:
- **Total Subjects:** 30 participants (IDs 1 through 30).
- **Training Set:** 17 subjects (`[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26]`) $\rightarrow$ **5,867 samples**.
- **Validation Set:** 4 subjects (`[27, 28, 29, 30]`) $\rightarrow$ **1,485 samples**.
- **Test Set:** 9 subjects (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`) $\rightarrow$ **2,947 samples**.
- **Subject Overlap:** $\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$. (Verified by automated unit tests).

### 3.2 Preprocessing
1. **StandardScaler:** Fit strictly on the 5,867 training windows ($\mu = 0.000, \sigma = 1.000$); applied to validation and test partitions without re-fitting.
2. **Principal Component Analysis (PCA):** 20 components chosen via validation elbow analysis, capturing **77.54%** of cumulative variance.

---

## 4. Empirical Benchmark Results

All experiments were executed locally on the benchmark environment with fixed seeds (`seed=42`).

### 4.1 Master Quantitative Comparison

| Experiment | Model Architecture | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Log Loss | Training Time (s) | Inference Latency (ms/sample) | False Switch Rate | Transition Consistency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B** | Logistic Regression | 0.9051 | 0.9107 | 0.8588 | 0.8576 | 0.3776 | 0.410 | 0.0001 | — | — |
| **B** | Decision Tree (CART) | 0.8000 | 0.7956 | 0.7669 | 0.7613 | 3.2189 | 0.060 | 0.0001 | — | — |
| **C** | Random Forest (Bagging) | 0.8808 | 0.8830 | 0.8395 | 0.8355 | 0.4340 | 0.434 | 0.0079 | — | — |
| **C** | HistGradientBoosting | 0.8721 | 0.8740 | 0.8463 | 0.8444 | 0.6884 | 6.919 | 0.0261 | — | — |
| **C** | Hard Voting (LR+RF+HGB) | 0.8889 | 0.8914 | 0.8578 | 0.8550 | — | 4.156 | 0.0367 | — | — |
| **C** | Soft Voting (LR+RF+HGB) | 0.8936 | 0.8973 | 0.8588 | 0.8565 | 0.3470 | 1.976 | 0.0368 | — | — |
| **C** | Stacking (Meta: LR) | **0.9091** | **0.9136** | **0.8653** | **0.8640** | 0.3688 | 3.313 | 0.0081 | — | — |
| **D** | HMM (Gaussian + BW) | — | — | 0.6939 | 0.6842 | — | 1.725 | 0.0039 | 0.0933 | 0.6306 |
| **E** | **Hybrid HMM-Ensemble** | — | — | **0.9362** | **0.9344** | — | — | 0.0385 | **0.0065** | **0.8018** |

### 4.2 Key Findings by Research Question

#### RQ1 & RQ2: Sequential vs. Independent Classifiers
- The standalone Gaussian HMM achieves 68.42% macro-F1, significantly lower than Logistic Regression (85.76%) and Stacking (86.40%). This confirms that fitting a full Gaussian distribution over a 20-dimensional PCA continuous space with a single mean and covariance per state is inadequate for complex dynamic movement.
- However, the HMM excels in sequential structure: the transition matrix correctly identifies that static postures (`SITTING`, `STANDING`, `LAYING`) have near-zero self-transition probabilities to dynamic walking states without intermediate postural transitions.

#### RQ3: Ensemble Methods Comparison
- Under identical PCA features and subject splits, Stacking achieves the highest standalone macro-F1 among ensembles (**86.40%**), outperforming single Random Forests (**83.55%**) and single Gradient Boosters (**84.44%**). Soft Voting achieved the lowest log loss (**0.3470**), confirming calibrated class probabilities.

#### RQ4: Hybrid HMM-Ensemble Synergy
- The Hybrid HMM-Ensemble achieves **93.44% macro-F1** and **93.62% accuracy**, an improvement of **+7.04% macro-F1 over the best ensemble** (Stacking: 86.40%) and **+25.02% over the HMM alone** (68.42%).
- Sequential evaluation reveals that the false activity switch rate plummets from **0.0933 down to 0.0065** — a **14.4-fold reduction** in erroneous classification flickers. Transition consistency reaches **80.18%**, accurately aligning with true physical activity boundaries.

### 4.3 Generalization to Unseen Participants (Experiment H)

| Test Subject ID | Random Forest Macro-F1 | Standalone HMM Macro-F1 | Dominant Performance Factor |
| :---: | :---: | :---: | :--- |
| **2** | 0.8855 | 0.7507 | High walking consistency |
| **4** | 0.8860 | 0.6916 | Standard postural transitions |
| **9** | 0.8266 | 0.5945 | Sitting/Standing overlap |
| **10** | 0.8127 | 0.7172 | High temporal regularity |
| **12** | 0.8329 | 0.6104 | Dynamic activity stability |
| **13** | 0.8608 | 0.6754 | Clear walking upstairs cadence |
| **18** | 0.7695 | 0.6334 | Irregular cadence |
| **20** | 0.7909 | 0.6805 | Sensor noise / mild tilt |
| **24** | 0.7955 | 0.6572 | Prolonged static posture |
| **Mean ± Std** | **0.8289 ± 0.041** | **0.6679 ± 0.048** | Zero subject leakage verified |

---

## 5. Visual Artifacts and Graphical Interpretations

The following high-resolution figures were generated directly from measured experimental outputs:

1. **Class Distribution (`A_class_distribution.png`):** Confirms balanced representation across all 6 activity classes in train, validation, and test splits with zero empty classes.
2. **PCA Scree Plot (`A_pca_explained_variance.png`):** Shows that 20 principal components explain 77.54% of dataset variance, reducing dimensionality by 96.4% while preserving discriminant signal power.
3. **Confusion Matrices (`B_cm_*.png`, `C_cm_*.png`, `D_cm_hmm.png`, `E_cm_hybrid.png`):** Illustrate that while baseline models frequently confuse `SITTING` and `STANDING`, the Hybrid model correctly disambiguates them using temporal transition context.
4. **HMM Transition Heatmap (`D_hmm_transition.png`):** Visualizes the empirical transition matrix $\mathbf{A}$, showing strong self-transition diagonals ($A_{ii} \approx 0.95-0.98$) and zero probability for physically impossible transitions (e.g., `WALKING_DOWNSTAIRS` $\rightarrow$ `LAYING`).
5. **Activity Timelines (`D_timeline_hmm.png`, `E_timeline_hybrid.png`):** Side-by-side ground truth vs. predicted timelines over consecutive windows, demonstrating how the hybrid removes isolated frame misclassifications.
6. **Inference Latency (`I_latency_comparison.png`):** Demonstrates that all models require under **0.04 ms per sample**, rendering them deployable in real-time embedded mobile applications (sampling at 50 Hz requires < 20 ms per window).
7. **Linear Dynamical System Kalman Filter (`LDS_kalman_filter.png`):** Compares raw triaxial accelerometer signals against continuous latent state estimates $\hat{z}_t$ with 95% confidence intervals, illustrating the continuous counterpart to discrete HMMs.

---

## 6. Interactive Web Application (Streamlit Demo)

To showcase the operational system, a comprehensive Streamlit web application is deployed at `streamlit_app/app.py`:
- **Page 1: Dataset Overview:** Displays sample distributions, PCA explained variance curves, and feature statistics.
- **Page 2: Live Activity Prediction:** Allows the user to select any unseen test subject and interactively view:
  - Real-time predicted activity timeline (Plotly interactive charts);
  - Side-by-side comparison of Ensemble vs. HMM vs. Hybrid predictions;
  - Confidence metric cards and normalized confusion matrices.
- **Page 3: Model Comparison:** Interactive multi-bar charts showing macro-F1, accuracy, and log-loss benchmarks.
- **Page 4: HMM Internals:** Interactive transition matrix heatmap, forward-backward state posterior area charts $\gamma_t(i)$, and Viterbi optimal path decoders.

**Execution Command:**
```bash
cd HAR_HMM_Project
streamlit run streamlit_app/app.py
```

---

## 7. Viva Voce Examination Questions and Detailed Answers

### Q1: Why does the Hybrid model outperform both the Ensemble and the HMM individually?
**Answer:** The ensemble (e.g., soft voting / stacking) is a highly discriminative model that learns complex non-linear decision boundaries over features, yielding accurate instantaneous posteriors $P(y_t \mid \mathbf{x}_t)$. However, it classifies each window in isolation and is susceptible to momentary sensor noise. The HMM provides a temporal regularizer via the transition matrix $\mathbf{A}$, enforcing temporal continuity. By dividing ensemble posteriors by class priors to obtain scaled likelihoods and feeding them into the Viterbi trellis, the hybrid combines the non-linear feature discrimination of ensembles with the temporal consistency of Markov chains, eliminating spurious classification flickers.

### Q2: What is the purpose of scaling factors $c_t$ in the Forward-Backward algorithm?
**Answer:** In the raw forward algorithm, $\alpha_t(i) = P(\mathbf{x}_1, \dots, \mathbf{x}_t, q_t = i \mid \lambda)$ is a joint probability of $t$ continuous observations. Because each emission density is typically $B_j(\mathbf{x}) < 1$, as sequence length $T$ grows ($T > 100$), $\alpha_t(i)$ decreases exponentially towards zero, causing floating-point underflow (evaluating to `0.0` in IEEE 754 precision). The scaling coefficient $c_t = \sum_i \alpha_t(i)$ renormalizes $\tilde{\alpha}_t(i) = \alpha_t(i) / c_t$ to sum to $1.0$ at every step. The total sequence log-likelihood is then recovered without underflow via $\ln P(\mathbf{X} \mid \lambda) = \sum_{t=1}^T \ln c_t$.

### Q3: How was data leakage prevented between training and evaluation?
**Answer:** We enforced a strict subject-independent split: the 30 participants were divided into disjoint groups (17 for training, 4 for validation, 9 for testing). No subject's data was ever shared across splits. Furthermore, the `StandardScaler` and `PCA` transformers were fit strictly on the training partition and only applied via `transform()` to the validation and test sets. Automated assertions in `tests/test_preprocessing.py` verify zero subject intersection.

### Q4: How does a Linear Dynamical System (LDS) relate to an HMM?
**Answer:** Both HMMs and LDS are state-space models sharing the same graphical structure. The fundamental difference lies in the latent state space: in an HMM, the latent state $q_t \in \{1, \dots, K\}$ is discrete; in an LDS (Kalman filter), the latent state $\mathbf{z}_t \in \mathbb{R}^d$ is continuous and evolves linearly with Gaussian noise ($\mathbf{z}_t = \mathbf{A}\mathbf{z}_{t-1} + \mathbf{w}_t$). Our project includes a Kalman filter demonstration in `src/kalman_filter.py` denoising raw triaxial accelerometer readings to showcase this continuous-state duality.

---

## 8. Conclusion and Future Work

This study has successfully developed, verified, and benchmarked a complete, CPU-friendly pipeline for Human Activity Recognition combining Hidden Markov Models (Unit 4) and Ensemble Learning (Unit 5). The resulting Hybrid HMM-Ensemble sets a strong benchmark at **93.44% test macro-F1** with a **14.4-fold reduction in false transitions**, running well within real-time mobile constraints (< 0.04 ms/sample). Future extensions include evaluating postural transitions on the UCI HAPT dataset and exploring higher-order semi-Markov models for explicit activity duration modelling.
