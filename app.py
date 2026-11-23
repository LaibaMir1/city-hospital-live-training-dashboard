"""
Project 5 - Live training dashboard
Client: City Hospital

Train a neural network on breast-tumour measurements, watch the learning
curves update live, fix underfitting/overfitting with the controls, then use
the finished model to screen held-out patients.

Run:  streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

import core

st.set_page_config(page_title="City Hospital - Live training dashboard",
                   page_icon="🩺", layout="wide")

# ---------------------------------------------------------------- data
@st.cache_data
def get_data():
    return core.load_splits()


X_train_full, y_train_full, X_val, y_val, X_test, y_test = get_data()

# ---------------------------------------------------------------- presets
PRESETS = {
    "Underfit": dict(frac=100, width=2, depth=1, alpha=0.0, lr=0.0005,
                     epochs=60, early=False, patience=15),
    "Overfit": dict(frac=15, width=256, depth=3, alpha=0.0, lr=0.01,
                    epochs=150, early=False, patience=15),
    "Fixed": dict(frac=100, width=64, depth=2, alpha=0.5, lr=0.003,
                  epochs=200, early=True, patience=15),
}
DEFAULTS = PRESETS["Overfit"]
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("runs", [])


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Training settings")
    st.caption("Start from a preset, or change the settings yourself.")
    for name in PRESETS:
        st.button(f"{name} preset", on_click=apply_preset, args=(name,),
                  width="stretch")

    st.subheader("Data")
    st.slider("Training patients used (%)", 5, 100, step=5, key="frac",
              help="Less data makes overfitting more likely.")

    st.subheader("Model size")
    st.select_slider("Neurons per layer", [2, 4, 8, 16, 32, 64, 128, 256],
                     key="width")
    st.slider("Hidden layers", 1, 4, key="depth")

    st.subheader("Training")
    st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.003, 0.01, 0.03],
                     key="lr")
    st.slider("Max epochs", 10, 300, step=10, key="epochs")
    st.select_slider("L2 regularisation (alpha)",
                     [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                     key="alpha", help="Penalises large weights. Higher = simpler model.")
    st.toggle("Early stopping", key="early",
              help="Stop when validation loss stops improving and keep the best weights.")
    st.slider("Patience (epochs)", 3, 40, key="patience",
              disabled=not st.session_state.early)

    st.subheader("Display")
    delay = st.slider("Animation delay (s per epoch)", 0.0, 0.2, 0.02, 0.01)

# ---------------------------------------------------------------- header
st.title("🩺 City Hospital: live training dashboard")
st.write(
    "A neural network learns to flag breast tumours that may be malignant from "
    "30 measurements of a cell biopsy. Train it, read the curves, fix them, "
    "then screen patients the model has never seen."
)
m1, m2, m3 = st.columns(3)
m1.metric("Training pool", f"{len(X_train_full)} patients")
m2.metric("Validation", f"{len(X_val)} patients")
m3.metric("Held-out new patients", f"{len(X_test)} patients")

tab_train, tab_screen, tab_about = st.tabs(
    ["1. Train and fix the curves", "2. Screen patients", "3. Limitations"])

# ================================================================ TAB 1
with tab_train:
    go = st.button("Train model", type="primary")
    status = st.empty()
    col_l, col_a = st.columns(2)
    col_l.markdown("**Loss** (lower is better)")
    loss_ph = col_l.empty()
    col_a.markdown("**Accuracy** (higher is better)")
    acc_ph = col_a.empty()
    verdict_ph = st.empty()

    def draw(hist):
        h = pd.DataFrame(hist).set_index("epoch")
        loss_ph.line_chart(h[["train_loss", "val_loss"]].rename(
            columns={"train_loss": "train", "val_loss": "validation"}),
            color=["#1f77b4", "#ff7f0e"])
        acc_ph.line_chart(h[["train_acc", "val_acc"]].rename(
            columns={"train_acc": "train", "val_acc": "validation"}),
            color=["#1f77b4", "#ff7f0e"])

    if go:
        ss = st.session_state
        X_tr, y_tr = core.subsample(X_train_full, y_train_full, ss.frac / 100)
        scaler = StandardScaler().fit(X_tr)
        model = core.make_model(ss.width, ss.depth, ss.alpha, ss.lr)
        hist = []
        for row in core.train_epochs(
                model, scaler.transform(X_tr), y_tr,
                scaler.transform(X_val), y_val,
                epochs=ss.epochs, patience=ss.patience if ss.early else None):
            hist.append(row)
            draw(hist)
            status.info(f"Epoch {row['epoch']}/{ss.epochs} · "
                        f"train loss {row['train_loss']:.3f} · "
                        f"val loss {row['val_loss']:.3f} · "
                        f"val acc {row['val_acc']:.1%}")
            if delay:
                time.sleep(delay)

        verdict, advice = core.diagnose(hist)
        last = hist[-1]
        msg = (f"Early stopping at epoch {last['epoch']}, best weights from "
               f"epoch {last['best_epoch']} restored." if last["stopped"]
               else f"Finished {last['epoch']} epochs.")
        status.success(msg)

        # Validation metrics for the model we kept
        _, val_acc, val_rec = core.epoch_metrics(model, scaler.transform(X_val), y_val)
        val_loss = min(r["val_loss"] for r in hist) if last["stopped"] else last["val_loss"]
        ss.model, ss.scaler, ss.hist = model, scaler, hist
        ss.verdict, ss.advice = verdict, advice
        ss.runs.append({
            "run": len(ss.runs) + 1, "diagnosis": verdict,
            "train patients": len(X_tr),
            "layers": f"{ss.depth} × {ss.width}", "lr": ss.lr, "L2": ss.alpha,
            "early stop": "yes" if ss.early else "no",
            "epochs run": last["epoch"],
            "val loss": round(val_loss, 3),
            "val accuracy": f"{val_acc:.1%}",
            "val sensitivity": f"{val_rec:.1%}",
        })
    elif "hist" in st.session_state:
        draw(st.session_state.hist)

    if "verdict" in st.session_state:
        box = {"Good fit": st.success, "Overfitting": st.warning,
               "Underfitting": st.error}[st.session_state.verdict]
        box(f"**Diagnosis: {st.session_state.verdict}.** {st.session_state.advice}")
    else:
        st.info("Pick a preset in the sidebar (try Overfit first) and press "
                "**Train model** to watch the curves.")

    if st.session_state.runs:
        st.markdown("**Run history**: compare your runs before and after each fix")
        st.dataframe(pd.DataFrame(st.session_state.runs).set_index("run"),
                     width="stretch")
        if st.button("Clear run history"):
            st.session_state.runs = []
            st.rerun()

# ================================================================ TAB 2
with tab_screen:
    if "model" not in st.session_state:
        st.info("Train a model in tab 1 first. Screening uses the last model you trained.")
    else:
        ss = st.session_state
        if ss.verdict != "Good fit":
            st.warning(f"The current model was diagnosed as **{ss.verdict.lower()}**. "
                       "Fix the curves before trusting its screening results.")

        st.subheader("Choose the alert threshold")
        st.write("A patient is flagged for follow-up when the predicted chance of "
                 "malignancy is at or above the threshold. Lower thresholds miss fewer "
                 "cancers but send more healthy patients for extra tests.")
        thr = st.slider("Flag patients at or above this probability", 0.05, 0.95,
                        0.30, 0.05)

        probs, flags = core.screen(ss.model, ss.scaler, X_test, thr)
        rep = core.screening_report(y_test, flags)

        st.subheader(f"Results on {len(X_test)} held-out patients")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sensitivity", f"{rep['sensitivity']:.1%}",
                  help="Share of malignant tumours that were flagged.")
        k2.metric("Specificity", f"{rep['specificity']:.1%}",
                  help="Share of benign tumours correctly not flagged.")
        k3.metric("Missed cancers", int(rep["fn"]),
                  help="Malignant cases the model did not flag. The costliest error.")
        k4.metric("False alarms", int(rep["fp"]))

        cm = pd.DataFrame(
            [[rep["tp"], rep["fn"]], [rep["fp"], rep["tn"]]],
            index=["Actually malignant", "Actually benign"],
            columns=["Flagged", "Not flagged"])
        st.table(cm)

        st.subheader("Screen one patient")
        pid = st.selectbox("Patient", X_test.index,
                           format_func=lambda i: f"Patient #{i}")
        pos = list(X_test.index).index(pid)
        p = probs[pos]
        truth = "malignant" if y_test[pos] == 1 else "benign"
        c1, c2 = st.columns([1, 2])
        c1.metric("Chance of malignancy", f"{p:.1%}")
        if p >= thr:
            c2.error(f"Flag for follow-up (threshold {thr:.0%}). Confirmed diagnosis: {truth}.")
        else:
            c2.success(f"No flag (threshold {thr:.0%}). Confirmed diagnosis: {truth}.")
        with st.expander("Show this patient's measurements"):
            st.dataframe(X_test.loc[[pid]].T.rename(columns={pid: "value"}),
                         width="stretch")

        st.subheader("Screen a batch from a CSV file")
        st.write("Upload a CSV with the same 30 measurement columns "
                 "(try `patients_to_screen.csv` from this repo).")
        up = st.file_uploader("Patient CSV", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            missing = [c for c in X_test.columns if c not in df.columns]
            if missing:
                st.error(f"The file is missing {len(missing)} required column(s), "
                         f"for example: {', '.join(missing[:3])}. "
                         "Use the column names from patients_to_screen.csv.")
            else:
                bp, bf = core.screen(ss.model, ss.scaler, df[X_test.columns], thr)
                out = df[[c for c in df.columns if c not in X_test.columns]].copy()
                out["chance_of_malignancy"] = np.round(bp, 3)
                out["flag_for_follow_up"] = np.where(bf == 1, "YES", "no")
                st.write(f"{int(bf.sum())} of {len(df)} patients flagged.")
                st.dataframe(out, width="stretch")
                st.download_button("Download results", out.to_csv(index=False),
                                   "screening_results.csv", "text/csv")

# ================================================================ TAB 3
with tab_about:
    st.subheader("Limitations")
    st.markdown("""
- **Not a medical device.** This is a teaching project. It must not be used to make decisions about real patients.
- **Small, old dataset.** 569 patients from one US hospital (Wisconsin, early 1990s). Results may not hold for other hospitals, scanners or populations.
- **Measurements, not images.** The model needs 30 numbers already measured from a biopsy image; it cannot read scans.
- **Small test set.** 114 held-out patients (42 malignant), so one extra missed cancer moves sensitivity by about 2.4 points.
- **Threshold is a clinical choice.** The default of 30% favours catching cancers over avoiding false alarms. A real hospital would set it with clinicians.
- **No uncertainty or explanation.** The model gives one probability with no confidence interval and no reason for its decision.
""")
    st.subheader("Data")
    st.markdown("Breast Cancer Wisconsin (Diagnostic) dataset, loaded from scikit-learn. "
                "Split: 60% training pool, 20% validation, 20% held-out new patients (fixed seed).")
