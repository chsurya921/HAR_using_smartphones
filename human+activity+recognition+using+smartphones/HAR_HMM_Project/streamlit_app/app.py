"""
streamlit_app/app.py
====================
Interactive website demo for the HAR HMM + Ensemble project.

Features
--------
- Upload a subject's data (or select from pre-loaded test subjects)
- Run: Baseline classifiers | HMM | Ensemble | Hybrid
- See: Activity timeline, class probabilities, confusion matrix
- Side-by-side comparison of HMM vs Ensemble vs Hybrid

Run with:
    cd HAR_HMM_Project
    streamlit run streamlit_app/app.py
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HAR: HMM + Ensemble Demo",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

ACTIVITY_COLORS = {
    1: "#2196F3",   # blue
    2: "#4CAF50",   # green
    3: "#FF9800",   # orange
    4: "#9C27B0",   # purple
    5: "#F44336",   # red
    6: "#00BCD4",   # cyan
}

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "experiment_config.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset …")
def load_data():
    pkl_path = os.path.join(PROJECT_DIR, "results", "demo_data.pkl")
    try:
        from data_loader import HARDataLoader
        from preprocessing import Preprocessor

        loader = HARDataLoader(config_path=CONFIG_PATH, base_dir=PROJECT_DIR)
        splits = loader.load_all()

        prep = Preprocessor(config_path=CONFIG_PATH)
        X_tr, X_va, X_te = prep.fit_transform(
            splits.train_X, splits.val_X, splits.test_X)
        seqs_tr = prep.transform_sequences(splits.sequences_train)
        seqs_te = prep.transform_sequences(splits.sequences_test)
        return splits, prep, X_tr, X_va, X_te, seqs_tr, seqs_te
    except Exception:
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, "rb") as fh:
                d = pickle.load(fh)
            class MockSplits:
                train_X = d["X_tr"]
                train_y = d["y_tr"]
                test_X = d["X_te"]
                test_y = d["y_te"]
                activity_labels = d["activity_labels"]
                feature_names = d["feature_names"]
            class MockPrep:
                n_features_out = d["X_tr"].shape[1]
                class MockPCA:
                    explained_variance_ratio_ = d["pca_variance"]
                pca = MockPCA() if d.get("pca_variance") is not None else None
            return MockSplits(), MockPrep(), d["X_tr"], d["X_te"][:10], d["X_te"], d["seqs_tr"], d["seqs_te"]
        raise


@st.cache_resource(show_spinner="Training models … (this may take ~60s)")
def train_models(X_tr, y_tr):
    from ensemble_models import build_all_models
    models = build_all_models(CONFIG_PATH)
    fitted = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
    return fitted


@st.cache_resource(show_spinner="Training HMM …")
def train_hmm(seqs_tr_list_X, seqs_tr_list_y):
    import yaml
    from hmm_model import GaussianHMM as MyHMM
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    model = MyHMM(
        n_states=cfg["hmm"]["n_states"],
        covariance_type=cfg["hmm"]["covariance_type"],
        n_iter=cfg["hmm"]["baum_welch"]["max_iter"],
        tol=cfg["hmm"]["baum_welch"]["tol"],
        min_variance=cfg["hmm"]["min_variance"],
        random_state=cfg["seeds"]["global"],
    )
    model._init_supervised(seqs_tr_list_X, seqs_tr_list_y)
    model.fit(seqs_tr_list_X)
    return model


@st.cache_resource(show_spinner="Building hybrid model …")
def build_hybrid(_hmm_model, _ensemble_model, seqs_tr_list_y, class_priors, n_classes):
    from hybrid_model import HybridHMMEnsemble
    A = np.exp(_hmm_model.log_A)
    hybrid = HybridHMMEnsemble(
        ensemble_model=_ensemble_model,
        transition_matrix=A,
        class_priors=class_priors,
        n_classes=n_classes,
    )
    hybrid.fit_transition_matrix(seqs_tr_list_y)
    return hybrid


# ─────────────────────────────────────────────────────────────────────────────
# Plotly timeline
# ─────────────────────────────────────────────────────────────────────────────

def plotly_timeline(y_true, y_pred, title="Activity Timeline"):
    """Interactive Plotly timeline comparing ground truth vs predictions."""
    t = np.arange(len(y_true))

    fig = go.Figure()

    # Ground truth
    fig.add_trace(go.Scatter(
        x=t, y=[ACTIVITY_LABELS[int(a)] for a in y_true],
        mode="lines", name="Ground Truth",
        line=dict(color="#2196F3", width=3),
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=t, y=[ACTIVITY_LABELS[int(a)] for a in y_pred],
        mode="lines", name="Predicted",
        line=dict(color="#F44336", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time window",
        yaxis_title="Activity",
        height=350,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig


def plotly_proba_bar(proba, title="Class Probabilities"):
    """Show mean class probabilities over the sequence."""
    mean_proba = proba.mean(axis=0)
    labels = [ACTIVITY_LABELS[i + 1] for i in range(len(mean_proba))]
    colors = [ACTIVITY_COLORS[i + 1] for i in range(len(mean_proba))]

    fig = go.Figure(go.Bar(
        x=labels,
        y=mean_proba,
        marker_color=colors,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Activity",
        yaxis_title="Mean probability",
        height=300,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig


def plotly_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(ACTIVITY_LABELS.values())
    cm = confusion_matrix(y_true, y_pred, labels=list(ACTIVITY_LABELS.keys()),
                          normalize="true")
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        text_auto=".2f",
        labels=dict(x="Predicted", y="True"),
        title="Confusion Matrix (normalised)",
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=40))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric summary card
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(col, label, value, delta=None):
    with col:
        st.metric(label=label, value=f"{value:.4f}", delta=f"{delta:+.4f}" if delta is not None else None)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.title("🏃 Human Activity Recognition")
    st.markdown(
        "**HMM + Ensemble Learning — Interactive Demo**  \n"
        "B.Tech / M.Tech (Integrated) CSE, SRMIST · Regulations 2021"
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")
        page = st.radio("Navigate", [
            "📊 Dataset Overview",
            "🤖 Live Prediction",
            "📈 Model Comparison",
            "🔬 HMM Internals",
        ])

    # ── Load data ─────────────────────────────────────────────────────────
    try:
        splits, prep, X_tr, X_va, X_te, seqs_tr, seqs_te = load_data()
    except Exception as e:
        st.error(
            f"❌ Could not load dataset: {e}\n\n"
            "Make sure the UCI HAR Dataset folder is at the correct path "
            "and `experiment_config.yaml` is present."
        )
        st.stop()

    y_tr = splits.train_y
    label_names = list(splits.activity_labels.values())

    # ── Page: Dataset Overview ─────────────────────────────────────────────
    if page == "📊 Dataset Overview":
        st.header("Dataset Overview — UCI HAR")

        col1, col2, col3 = st.columns(3)
        col1.metric("Training samples", f"{len(y_tr):,}")
        col2.metric("Test samples", f"{len(splits.test_y):,}")
        col3.metric("Features (raw)", "561 → PCA")

        col4, col5, col6 = st.columns(3)
        col4.metric("Subjects (total)", "30")
        col5.metric("Activities", "6")
        col6.metric(f"PCA components", str(prep.n_features_out))

        st.subheader("Class Distribution")
        train_counts = pd.Series(y_tr).map(ACTIVITY_LABELS).value_counts().reset_index()
        train_counts.columns = ["Activity", "Count"]
        fig = px.bar(train_counts, x="Activity", y="Count",
                     color="Activity", title="Training set class distribution")
        st.plotly_chart(fig, width='stretch')

        if prep.pca is not None:
            st.subheader("PCA Explained Variance")
            cumvar = prep.pca.explained_variance_ratio_.cumsum()
            df_pca = pd.DataFrame({"Component": range(1, len(cumvar) + 1),
                                    "Cumulative Variance Explained": cumvar})
            fig2 = px.line(df_pca, x="Component", y="Cumulative Variance Explained",
                           markers=True, title="PCA cumulative explained variance")
            fig2.add_hline(y=0.95, line_dash="dash", annotation_text="95%")
            st.plotly_chart(fig2, width="stretch")

    # ── Page: Live Prediction ──────────────────────────────────────────────
    elif page == "🤖 Live Prediction":
        st.header("Live Activity Prediction")

        subject_ids = sorted({seq.subject_id for seq in seqs_te})
        sel_subj = st.selectbox("Select a test subject:", subject_ids)

        seq = next(s for s in seqs_te if s.subject_id == sel_subj)
        models_fitted = train_models(X_tr, y_tr)
        hmm_model = train_hmm(
            [s.X for s in seqs_tr], [s.y for s in seqs_tr])

        unique, counts = np.unique(y_tr, return_counts=True)
        class_priors = counts / counts.sum()

        hybrid = build_hybrid(
            hmm_model,
            models_fitted["soft_voting"],
            [s.y for s in seqs_tr],
            class_priors,
            n_classes=6,
        )

        from sklearn.metrics import f1_score, accuracy_score

        # Ensemble prediction
        ens_pred  = models_fitted["soft_voting"].predict(seq.X)
        ens_proba = models_fitted["soft_voting"].predict_proba(seq.X)

        # HMM prediction
        hmm_pred  = hmm_model.predict(seq.X) + 1
        hmm_proba = hmm_model.predict_proba(seq.X)

        # Hybrid prediction
        hyb_pred  = hybrid.predict(seq.X, ens_proba)

        st.subheader(f"Subject {sel_subj} — Predictions")

        m1, m2, m3 = st.columns(3)
        m1.metric("Ensemble macro-F1",
                  f"{f1_score(seq.y, ens_pred, average='macro', zero_division=0):.4f}")
        m2.metric("HMM macro-F1",
                  f"{f1_score(seq.y, hmm_pred, average='macro', zero_division=0):.4f}")
        m3.metric("Hybrid macro-F1",
                  f"{f1_score(seq.y, hyb_pred, average='macro', zero_division=0):.4f}")

        st.plotly_chart(plotly_timeline(seq.y, ens_pred, "Ensemble Predictions"), width="stretch")
        st.plotly_chart(plotly_timeline(seq.y, hmm_pred, "HMM Predictions"), width="stretch")
        st.plotly_chart(plotly_timeline(seq.y, hyb_pred, "Hybrid Predictions"), width="stretch")

        st.subheader("Class Probabilities (Ensemble)")
        st.plotly_chart(plotly_proba_bar(ens_proba, "Mean class probabilities — Ensemble"),
                        width="stretch")

        st.subheader("Confusion Matrix — Ensemble")
        st.plotly_chart(plotly_confusion(seq.y, ens_pred), width="stretch")

    # ── Page: Model Comparison ────────────────────────────────────────────
    elif page == "📈 Model Comparison":
        st.header("Model Comparison — All Methods")

        metrics_path = os.path.join(PROJECT_DIR, "results", "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            if "test_macro_f1" in df.columns and "model" in df.columns:
                fig = px.bar(df.dropna(subset=["test_macro_f1"]),
                             x="model", y="test_macro_f1",
                             color="experiment", barmode="group",
                             title="Test set macro-F1 by model and experiment")
                st.plotly_chart(fig, width='stretch')
                st.dataframe(df)
            else:
                st.info("Run `python src/experiment_runner.py` to populate results.")
        else:
            st.info("No results found yet. Run `python src/experiment_runner.py` first.")

    # ── Page: HMM Internals ───────────────────────────────────────────────
    elif page == "🔬 HMM Internals":
        st.header("HMM Internals")

        hmm_model = train_hmm(
            [s.X for s in seqs_tr], [s.y for s in seqs_tr])

        st.subheader("Transition Matrix A")
        A = np.exp(hmm_model.log_A)
        df_A = pd.DataFrame(A,
                            index=[f"From: {ACTIVITY_LABELS[i+1]}" for i in range(6)],
                            columns=[f"To: {ACTIVITY_LABELS[i+1]}" for i in range(6)])
        fig_A = px.imshow(A,
                          x=label_names, y=label_names,
                          color_continuous_scale="Blues",
                          text_auto=".3f",
                          title="HMM Transition Matrix A_ij = P(q_{t+1}=j | q_t=i)")
        st.plotly_chart(fig_A, width="stretch")

        st.subheader("State Posteriors for a Test Subject")
        sel_subj = st.selectbox("Subject:", sorted({s.subject_id for s in seqs_te}))
        seq = next(s for s in seqs_te if s.subject_id == sel_subj)
        gamma = hmm_model.predict_proba(seq.X)   # (T, 6)
        df_g = pd.DataFrame(gamma, columns=label_names)
        fig_g = px.area(df_g, title=f"State posteriors γ_t(i) — Subject {sel_subj}")
        st.plotly_chart(fig_g, width="stretch")

        st.subheader("Viterbi Decoded Path")
        path = hmm_model.predict(seq.X) + 1
        st.plotly_chart(
            plotly_timeline(seq.y, path, f"Viterbi path — Subject {sel_subj}"),
            width="stretch",
        )

        st.info(
            "**Mathematical background:**  \n"
            "The HMM is defined by λ = (π, A, B).  \n"
            "- **π** = initial state distribution  \n"
            "- **A** = transition matrix (shown above)  \n"
            "- **B** = Gaussian emission model per state  \n\n"
            "Posterior state probabilities are computed using the **Forward-Backward** "
            "algorithm. The most likely path is decoded with **Viterbi** "
            "(log-space dynamic programming)."
        )

    st.markdown("---")
    st.markdown("*HAR HMM + Ensemble project · SRMIST CSE · 2024*")


if __name__ == "__main__":
    main()
