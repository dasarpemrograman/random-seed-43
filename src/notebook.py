import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FFNN
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    from ffnn.model import FFNN

    return FFNN, LabelEncoder, StandardScaler, np, pd, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("src/data/datasetml_2026.csv")

    print("Shape:")
    print(df.shape)

    print("\nTypes:")
    print(df.dtypes)

    print("\nHead:")
    print(df.head())
    return (df,)


@app.cell
def _(df):
    print("Missing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["placement_status"].value_counts())
    return


@app.cell
def _():
    num_cols = [
        "cgpa",
        "backlogs",
        "internship_count",
        "aptitude_score",
        "communication_score",
        "internship_quality_score",
    ]
    cat_cols = [
        "college_tier",
        "country",
        "university_ranking_band",
        "specialization",
        "industry",
    ]
    return cat_cols, num_cols


@app.cell
def _(cat_cols, df, pd):
    df_encoded = pd.get_dummies(
        df.copy(),
        columns=cat_cols,
        drop_first=True,
        dtype=float,
    )

    print(df_encoded.shape)
    print(df_encoded.columns.tolist())
    return (df_encoded,)


@app.cell
def _(LabelEncoder, df_encoded, np):
    le = LabelEncoder()
    y = (
        le.fit_transform(df_encoded["placement_status"])
        .reshape(-1, 1)
        .astype(np.float64)
    )

    print("Target classes: ")
    print(le.classes_)

    print("\nTarget shape:")
    print(y.shape)

    print("\nSample values: ")
    print(y[:5])
    return (y,)


@app.cell
def _(df_encoded, np):
    feature_cols = [col for col in df_encoded.columns if col != "placement_status"]
    X = df_encoded[feature_cols].values.astype(np.float64)

    print(X.shape)
    print(feature_cols)
    return X, feature_cols


@app.cell
def _(X, train_test_split, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}  y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}  y_test  : {y_test.shape}")
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(X_train, y_train):
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train.ravel())
    y_train_sm = y_train_sm.reshape(-1, 1)

    print(f"  Before SMOTE : {X_train.shape}  {y_train.shape}")
    print(f"  After  SMOTE : {X_train_sm.shape}  {y_train_sm.shape}")
    return X_train_sm, y_train_sm


@app.cell
def _(
    StandardScaler,
    X_test,
    X_train_sm,
    X_val,
    feature_cols,
    np,
    num_cols,
    y_test,
    y_train_sm,
    y_val,
):
    idx_num = [i for i, col in enumerate(feature_cols) if col in num_cols]

    scaler = StandardScaler()

    X_train_final = X_train_sm.copy()
    X_val_final = X_val.copy()
    X_test_final = X_test.copy()

    X_train_final[:, idx_num] = scaler.fit_transform(X_train_sm[:, idx_num])
    X_val_final[:, idx_num] = scaler.transform(X_val[:, idx_num])
    X_test_final[:, idx_num] = scaler.transform(X_test[:, idx_num])

    X_train_final = X_train_final.astype(np.float64)
    X_val_final = X_val_final.astype(np.float64)
    X_test_final = X_test_final.astype(np.float64)

    y_train_final = y_train_sm.astype(np.float64)
    y_val_final = y_val.astype(np.float64)
    y_test_final = y_test.astype(np.float64)

    print(f"  X_train : {X_train_final.shape}  y_train : {y_train_final.shape}")
    print(f"  X_val   : {X_val_final.shape}  y_val   : {y_val_final.shape}")
    print(f"  X_test  : {X_test_final.shape}  y_test  : {y_test_final.shape}")
    return (
        X_test_final,
        X_train_final,
        X_val_final,
        y_test_final,
        y_train_final,
        y_val_final,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiments
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(FFNN, X_train_final, X_val_final, y_train_final, y_val_final):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    def build_and_train(
        hidden_sizes,
        activations_hidden,
        *,
        lr=0.01,
        epochs=100,
        batch_size=32,
        initializer="uniform",
        regularizer="none",
        reg_kwargs=None,
    ):
        n_in = X_train_final.shape[1]
        layer_sizes = [n_in] + list(hidden_sizes) + [1]
        activations = list(activations_hidden) + ["sigmoid"]

        model = FFNN(
            layer_sizes=layer_sizes,
            activations=activations,
            loss="bce",
            initializer=initializer,
            regularizer=regularizer,
            reg_kwargs=reg_kwargs or {},
        )
        history = model.fit(
            X_train_final,
            y_train_final,
            batch_size=batch_size,
            learning_rate=lr,
            epochs=epochs,
            verbose=0,
            validation_data=(X_val_final, y_val_final),
        )
        return model, history

    def eval_model(model, X, y_true):
        y_pred = model.predict(X)
        y_flat = y_true.flatten().astype(int)
        kw = dict(average="weighted", zero_division=0)
        return {
            "Accuracy": round(accuracy_score(y_flat, y_pred), 4),
            "F1": round(f1_score(y_flat, y_pred, **kw), 4),
            "Precision": round(precision_score(y_flat, y_pred, **kw), 4),
            "Recall": round(recall_score(y_flat, y_pred, **kw), 4),
        }

    def refresh_gradients(model, X, y):
        y_pred = model.forward(X)
        model.backward(y, y_pred)

    def plot_loss_curves(histories, labels, title="Loss Curves"):
        colors = plt.cm.tab10.colors
        fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for i, (hist, lbl) in enumerate(zip(histories, labels)):
            c = colors[i % len(colors)]
            ax_tr.plot(hist["train_loss"], label=lbl, color=c)
            if hist["val_loss"]:
                ax_val.plot(hist["val_loss"], label=lbl, color=c)

        for ax, name in zip((ax_tr, ax_val), ("Train Loss", "Val Loss")):
            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_wgrad_dists(
        models, labels, layer_idx, title="Weight & Gradient Distributions"
    ):
        import numpy as _np

        n = len(models)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        if n == 1:
            axes = axes.reshape(2, 1)

        for j, (model, lbl) in enumerate(zip(models, labels)):
            W = model.layers[layer_idx].W.flatten()
            dW = model.layers[layer_idx].dW.flatten()

            axes[0, j].hist(W, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
            axes[0, j].set_title(f"Weights\n{lbl}", fontsize=9)
            axes[0, j].set_xlabel("Value")
            axes[0, j].set_ylabel("Count")

            axes[1, j].hist(dW, bins=60, color="tomato", alpha=0.8, edgecolor="none")
            axes[1, j].set_title(f"Gradients\n{lbl}", fontsize=9)
            axes[1, j].set_xlabel("Value")
            axes[1, j].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    return (
        accuracy_score,
        build_and_train,
        eval_model,
        f1_score,
        plot_loss_curves,
        plot_wgrad_dists,
        precision_score,
        recall_score,
        refresh_gradients,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 1a – Width Variation (depth fixed at 2 hidden layers)

    | Config | Hidden layers |
    |--------|--------------|
    | A | [16, 16] |
    | B | [64, 64] |
    | C | [256, 256] |

    Fixed: `lr=0.01`, `epochs=100`, `batch_size=32`, `activation=relu`, `init=uniform`
    """)
    return


@app.cell
def _(
    X_test_final,
    build_and_train,
    eval_model,
    pd,
    plot_loss_curves,
    y_test_final,
):
    _configs_1a = {
        "A: [16, 16]": [16, 16],
        "B: [64, 64]": [64, 64],
        "C: [256, 256]": [256, 256],
    }

    _histories_1a, _rows_1a = [], []

    for _lbl, _hidden in _configs_1a.items():
        _model, _hist = build_and_train(
            _hidden,
            ["relu"] * len(_hidden),
            lr=0.01,
            epochs=100,
            batch_size=32,
        )
        _metrics = eval_model(_model, X_test_final, y_test_final)
        _histories_1a.append(_hist)
        _rows_1a.append({"Config": _lbl, **_metrics})

    plot_loss_curves(
        _histories_1a,
        list(_configs_1a.keys()),
        "Exp 1a: Width Variation – Train / Val Loss",
    )

    _df_1a = pd.DataFrame(_rows_1a).set_index("Config")
    print("\nExp 1a – Final Metrics")
    print(_df_1a.to_string())
    _df_1a
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 1b – Depth Variation (width fixed at 64 per layer)

    | Config | Hidden layers |
    |--------|--------------|
    | A | [64] |
    | B | [64, 64, 64] |
    | C | [64, 64, 64, 64, 64] |

    Fixed: `lr=0.01`, `epochs=100`, `batch_size=32`, `activation=relu`, `init=uniform`
    """)
    return


@app.cell
def _(
    X_test_final,
    build_and_train,
    eval_model,
    pd,
    plot_loss_curves,
    y_test_final,
):
    _configs_1b = {
        "A: [64]": [64],
        "B: [64]*3": [64, 64, 64],
        "C: [64]*5": [64, 64, 64, 64, 64],
    }

    _histories_1b, _rows_1b = [], []

    for _lbl, _hidden in _configs_1b.items():
        _model, _hist = build_and_train(
            _hidden,
            ["relu"] * len(_hidden),
            lr=0.01,
            epochs=100,
            batch_size=32,
        )
        _metrics = eval_model(_model, X_test_final, y_test_final)
        _histories_1b.append(_hist)
        _rows_1b.append({"Config": _lbl, **_metrics})

    plot_loss_curves(
        _histories_1b,
        list(_configs_1b.keys()),
        "Exp 1b: Depth Variation – Train / Val Loss",
    )

    _df_1b = pd.DataFrame(_rows_1b).set_index("Config")
    print("\nExp 1b – Final Metrics")
    print(_df_1b.to_string())
    _df_1b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 2 – Activation Function Variation

    Architecture: `input → 64 (relu) → 64 (TEST) → 64 (relu) → output (sigmoid)`

    The **test layer** (`model.layers[1]`, the middle hidden layer) cycles through:
    `linear`, `relu`, `sigmoid`, `tanh`. All other layers are fixed to ReLU.

    Plots:
    - Train / Val loss curves for all 4 models
    - Weight distribution of the test layer after training
    - Gradient distribution of the test layer after training
    """)
    return


@app.cell
def _(
    X_test_final,
    X_train_final,
    build_and_train,
    eval_model,
    pd,
    plot_loss_curves,
    plot_wgrad_dists,
    refresh_gradients,
    y_test_final,
    y_train_final,
):
    _TEST_LAYER_IDX = 1
    _test_acts = ["linear", "relu", "sigmoid", "tanh"]

    _histories_2, _models_2, _labels_2, _rows_2 = [], [], [], []

    for _act in _test_acts:
        _model, _hist = build_and_train(
            [64, 64, 64],
            ["relu", _act, "relu"],
            lr=0.01,
            epochs=100,
            batch_size=32,
        )
        refresh_gradients(_model, X_train_final, y_train_final)
        _metrics = eval_model(_model, X_test_final, y_test_final)
        _histories_2.append(_hist)
        _models_2.append(_model)
        _labels_2.append(f"Test: {_act}")
        _rows_2.append({"Activation": _act, **_metrics})

    plot_loss_curves(
        _histories_2,
        _labels_2,
        "Exp 2: Activation Variation – Train / Val Loss",
    )
    plot_wgrad_dists(
        _models_2,
        _labels_2,
        _TEST_LAYER_IDX,
        "Exp 2: Test-Layer Weight & Gradient Distributions",
    )

    _df_2 = pd.DataFrame(_rows_2).set_index("Activation")
    print("\nExp 2 – Final Metrics")
    print(_df_2.to_string())
    _df_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 3 – Learning Rate Variation

    Architecture: `input → 64 (relu) → 64 (relu) → output (sigmoid)`

    | Run | LR |
    |-----|----|
    | A | 0.001 |
    | B | 0.01 |
    | C | 0.1 |

    > **Note:** LR C (0.1) may cause the loss to diverge or oscillate significantly.
    > This is expected behaviour and is visible in the curves below.

    Plots: loss curves + weight & gradient distributions for `model.layers[1]`.
    """)
    return


@app.cell
def _(
    X_test_final,
    X_train_final,
    build_and_train,
    eval_model,
    pd,
    plot_loss_curves,
    plot_wgrad_dists,
    refresh_gradients,
    y_test_final,
    y_train_final,
):
    _VIS_LAYER_IDX_3 = 1
    _lrs = {"LR A: 0.001": 0.001, "LR B: 0.01": 0.01, "LR C: 0.1": 0.1}

    _histories_3, _models_3, _labels_3, _rows_3 = [], [], [], []

    for _lbl, _lr in _lrs.items():
        _model, _hist = build_and_train(
            [64, 64],
            ["relu", "relu"],
            lr=_lr,
            epochs=100,
            batch_size=32,
        )
        refresh_gradients(_model, X_train_final, y_train_final)
        _metrics = eval_model(_model, X_test_final, y_test_final)
        _histories_3.append(_hist)
        _models_3.append(_model)
        _labels_3.append(_lbl)
        _rows_3.append({"LR config": _lbl, **_metrics})

    plot_loss_curves(
        _histories_3,
        _labels_3,
        "Exp 3: LR Variation – Train / Val Loss",
    )
    plot_wgrad_dists(
        _models_3,
        _labels_3,
        _VIS_LAYER_IDX_3,
        "Exp 3: Weight & Gradient Distributions (model.layers[1])",
    )

    _df_3 = pd.DataFrame(_rows_3).set_index("LR config")
    print("\nExp 3 – Final Metrics")
    print(_df_3.to_string())
    _df_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 4 – Regularization

    Arsitektur: `input → 64 (relu) → 64 (relu) → output (sigmoid)`
    Fixed: `lr=0.01`, `epochs=100`, `batch_size=32`

    | Model | Regularizer |
    |-------|-------------|
    | A | None |
    | B | L1, λ=0.001 |
    | C | L2, λ=0.001 |

    Expected observations:
    - **Model A**: no penalty — possible overfitting (train loss well below val loss)
    - **Model B (L1)**: sparse weights (many values near zero)
    - **Model C (L2)**: small but non-zero weights (smoother distribution)
    """)
    return


@app.cell
def _(
    X_test_final,
    X_train_final,
    build_and_train,
    eval_model,
    pd,
    plot_loss_curves,
    plot_wgrad_dists,
    refresh_gradients,
    y_test_final,
    y_train_final,
):
    _VIS_LAYER_IDX_4 = 1
    _reg_configs = {
        "A: No Reg": dict(regularizer="none", reg_kwargs={}),
        "B: L1 (0.001)": dict(regularizer="l1", reg_kwargs={"lambda_": 0.001}),
        "C: L2 (0.001)": dict(regularizer="l2", reg_kwargs={"lambda_": 0.001}),
    }

    _histories_4, _models_4, _labels_4, _rows_4 = [], [], [], []

    for _lbl, _cfg in _reg_configs.items():
        _model, _hist = build_and_train(
            [64, 64],
            ["relu", "relu"],
            lr=0.01,
            epochs=100,
            batch_size=32,
            regularizer=_cfg["regularizer"],
            reg_kwargs=_cfg["reg_kwargs"],
        )
        refresh_gradients(_model, X_train_final, y_train_final)
        _metrics = eval_model(_model, X_test_final, y_test_final)
        _histories_4.append(_hist)
        _models_4.append(_model)
        _labels_4.append(_lbl)
        _rows_4.append({"Regularizer": _lbl, **_metrics})

    plot_loss_curves(
        _histories_4,
        _labels_4,
        "Exp 4: Regularization – Train / Val Loss",
    )
    plot_wgrad_dists(
        _models_4,
        _labels_4,
        _VIS_LAYER_IDX_4,
        "Exp 4: Weight & Gradient Distributions (model.layers[1])",
    )

    _df_4 = pd.DataFrame(_rows_4).set_index("Regularizer")
    print("\nExp 4 – Final Metrics")
    print(_df_4.to_string())
    _df_4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 5 – Comparison with sklearn MLPClassifier

    Fixed config: hidden=[64, 64], activation=relu, lr=0.01, epochs=100, batch_size=32.

    **Fairness caveats**
    - `solver='sgd'` is chosen so sklearn uses vanilla SGD, matching our update rule.
    - `learning_rate='constant'` disables sklearn's default adaptive schedule.
    - sklearn may differ in mini-batch shuffling, weight initialisation (Glorot uniform),
      and numerical precision of the loss gradient.
    - Both models are evaluated on identical scaled test features.
    """)
    return


@app.cell
def _(
    X_test_final,
    X_train_final,
    accuracy_score,
    build_and_train,
    eval_model,
    f1_score,
    pd,
    precision_score,
    recall_score,
    y_test_final,
    y_train_final,
):
    from sklearn.neural_network import MLPClassifier

    _scratch, _our_hist = build_and_train(
        [64, 64],
        ["relu", "relu"],
        lr=0.01,
        epochs=100,
        batch_size=32,
    )
    _scratch_metrics = eval_model(_scratch, X_test_final, y_test_final)

    _sk = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=0.01,
        max_iter=100,
        batch_size=32,
        random_state=42,
    )
    _sk.fit(X_train_final, y_train_final.ravel())
    _sk_pred = _sk.predict(X_test_final)
    _y_true = y_test_final.flatten().astype(int)
    _kw = dict(average="weighted", zero_division=0)
    _sk_metrics = {
        "Accuracy": round(accuracy_score(_y_true, _sk_pred), 4),
        "F1": round(f1_score(_y_true, _sk_pred, **_kw), 4),
        "Precision": round(precision_score(_y_true, _sk_pred, **_kw), 4),
        "Recall": round(recall_score(_y_true, _sk_pred, **_kw), 4),
    }

    _df_5 = pd.DataFrame(
        [
            {"Model": "From scratch FFNN", **_scratch_metrics},
            {"Model": "sklearn MLP", **_sk_metrics},
        ]
    ).set_index("Model")

    print("\nExp 5 – FFNN vs sklearn MLPClassifier")
    print(_df_5.to_string())
    _df_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exp 6 – Initializer Comparison

    Architecture: `input → 64 (relu) → 64 (relu) → output (sigmoid)`
    Fixed: `lr=0.01`, `epochs=100`, `batch_size=32`

    | Config | Initializer |
    |--------|-------------|
    | A | Uniform |
    | B | Xavier |
    | C | He |
    """)
    return


@app.cell
def _(
    X_test_final,
    build_and_train,
    eval_model,
    pd,
    plot_loss_curves,
    y_test_final,
):
    _configs_6 = {
        "A: Uniform": "uniform",
        "B: Xavier": "xavier",
        "C: He": "he",
    }

    _histories_6, _rows_6 = [], []

    for _lbl, _init in _configs_6.items():
        _model, _hist = build_and_train(
            [64, 64],
            ["relu", "relu"],
            lr=0.01,
            epochs=100,
            batch_size=32,
            initializer=_init,
        )
        _metrics = eval_model(_model, X_test_final, y_test_final)
        _histories_6.append(_hist)
        _rows_6.append({"Config": _lbl, **_metrics})

    plot_loss_curves(
        _histories_6,
        list(_configs_6.keys()),
        "Exp 6: Initializer Comparison – Train / Val Loss",
    )

    _df_6 = pd.DataFrame(_rows_6).set_index("Config")
    print("\nExp 6 – Final Metrics")
    print(_df_6.to_string())
    _df_6
    return


if __name__ == "__main__":
    app.run()
