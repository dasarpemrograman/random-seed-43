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

    return LabelEncoder, StandardScaler, np, pd, train_test_split


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
def _(
    StandardScaler,
    X_test,
    X_train,
    X_val,
    feature_cols,
    np,
    num_cols,
    y_test,
    y_train,
    y_val,
):
    idx_num = [
        i for i, col in enumerate(feature_cols) if col in num_cols
    ]

    scaler = StandardScaler()

    X_train_final = X_train.copy()
    X_val_final = X_val.copy()
    X_test_final = X_test.copy()

    X_train_final[:, idx_num] = scaler.fit_transform(
        X_train[:, idx_num]
    )
    X_val_final[:, idx_num] = scaler.transform(X_val[:, idx_num])
    X_test_final[:, idx_num] = scaler.transform(X_test[:, idx_num])


    X_train_final = X_train_final.astype(np.float64)
    X_val_final = X_val_final.astype(np.float64)
    X_test_final = X_test_final.astype(np.float64)

    y_train_final = y_train.astype(np.float64)
    y_val_final = y_val.astype(np.float64)
    y_test_final = y_test.astype(np.float64)

    print(f"  X_train : {X_train_final.shape}  y_train : {y_train_final.shape}")
    print(f"  X_val   : {X_val_final.shape}  y_val   : {y_val_final.shape}")
    print(f"  X_test  : {X_test_final.shape}  y_test  : {y_test_final.shape}")
    return


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


if __name__ == "__main__":
    app.run()
