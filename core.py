"""Training logic for the live dashboard, kept separate from the UI so it can be tested."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, recall_score, confusion_matrix

SEED = 7
CLASSES = np.array([0, 1])   # 1 = malignant (needs follow-up), 0 = benign


def load_splits():
    """60% train pool / 20% validation / 20% held-out 'new patients'."""
    data = load_breast_cancer(as_frame=True)
    X = data.frame[data.feature_names]
    y = 1 - data.target.values            # sklearn uses 0 = malignant; flip it
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=0.25, stratify=y_pool, random_state=SEED)
    return X_train, y_train, X_val, y_val, X_test, y_test


def subsample(X, y, fraction, seed=SEED):
    """Use only part of the training data (to show what little data does)."""
    if fraction >= 1.0:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=fraction, stratify=y,
                                    random_state=seed)
    return Xs, ys


def make_model(width, depth, alpha, lr, seed=SEED):
    return MLPClassifier(hidden_layer_sizes=(width,) * depth, alpha=alpha,
                         learning_rate_init=lr, random_state=seed)


def epoch_metrics(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return (log_loss(y, p, labels=CLASSES),
            accuracy_score(y, p >= 0.5),
            recall_score(y, p >= 0.5, zero_division=0))


def train_epochs(model, X_tr, y_tr, X_va, y_va, epochs, patience=None, seed=SEED):
    """Generator: trains one epoch at a time and yields the metrics so the UI
    can redraw the curves live. With patience set, stops early and restores
    the best weights."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None, 0)
    wait = 0
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(X_tr))
        model.partial_fit(X_tr[idx], y_tr[idx], classes=CLASSES)
        tl, ta, _ = epoch_metrics(model, X_tr, y_tr)
        vl, va, vr = epoch_metrics(model, X_va, y_va)
        stop = False
        if vl < best[0] - 1e-4:
            best = (vl, ([w.copy() for w in model.coefs_],
                         [b.copy() for b in model.intercepts_]), epoch)
            wait = 0
        else:
            wait += 1
            if patience is not None and wait >= patience:
                model.coefs_, model.intercepts_ = best[1]
                stop = True
        yield dict(epoch=epoch, train_loss=tl, val_loss=vl, train_acc=ta,
                   val_acc=va, val_recall=vr, best_epoch=best[2], stopped=stop)
        if stop:
            break


def diagnose(hist):
    """Read the curves the way we did in class and return (verdict, advice)."""
    h = pd.DataFrame(hist)
    best_vl = h.val_loss.min()
    best_ep = int(h.val_loss.idxmin()) + 1
    early_stopped = bool(h.stopped.iloc[-1])
    # With early stopping the model was rolled back to its best epoch,
    # so judge the model we actually kept.
    last = h.iloc[best_ep - 1] if early_stopped else h.iloc[-1]
    gap = last.train_acc - last.val_acc
    rising = (not early_stopped) and last.val_loss > best_vl * 1.10 \
        and best_ep < len(h) * 0.8

    if last.train_acc < 0.93 and last.val_acc < 0.93:
        return ("Underfitting",
                "Both curves are still poor, so the model has not learned the pattern. "
                "Make the network wider or deeper, raise the learning rate, train for more "
                "epochs, or lower L2.")
    if rising or gap > 0.04:
        return ("Overfitting",
                f"Training keeps improving but validation loss was best at epoch {best_ep} "
                f"and the accuracy gap is {gap:.1%}. Add L2, turn on early stopping, "
                "use more training data, or make the network smaller.")
    return ("Good fit",
            "Train and validation curves flatten out together with a small gap. "
            "This model is ready to screen new patients.")


def screen(model, scaler, X, threshold):
    p = model.predict_proba(scaler.transform(X))[:, 1]
    return p, (p >= threshold).astype(int)


def screening_report(y_true, flags):
    tn, fp, fn, tp = confusion_matrix(y_true, flags, labels=[0, 1]).ravel()
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                sensitivity=tp / (tp + fn) if tp + fn else 0.0,
                specificity=tn / (tn + fp) if tn + fp else 0.0,
                accuracy=(tp + tn) / (tp + tn + fp + fn))
