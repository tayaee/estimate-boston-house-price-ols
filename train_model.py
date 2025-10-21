import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ut_model import save_model_and_metadata

TARGET_NAME = "MEDV"
VIF_THRESHOLD = 5.0
P_VALUE_THRESHOLD = 0.05
RANDOM_STATE = 42


def load_and_prepare_data(input_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads Boston Housing data from CSV and prepares it."""
    print(f"Loading data from {input_path}...")

    # 1. Load data from CSV
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise ValueError("Failed to load data. Please check the path.")

    # 2. Handle CHAS categorical variable
    if "CHAS" in df.columns:
        df["CHAS"] = df["CHAS"].astype(int).replace({1: "yes", 0: "no"})

    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME]

    # 3. One-hot encode categorical variables
    X = pd.get_dummies(
        X,
        columns=X.select_dtypes(include=["object", "category"]).columns.tolist(),
        drop_first=True,
        dtype=float,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def check_vif_pvalue_ols(
    X_data: pd.DataFrame,
    y_data: pd.Series,
) -> tuple[RegressionResultsWrapper, pd.DataFrame, pd.Series]:
    """Fits OLS model, returns VIF and P-values."""
    X_sm: pd.DataFrame = sm.add_constant(X_data)  # type: ignore
    model: RegressionResultsWrapper = sm.OLS(y_data, X_sm).fit()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_data.columns
    vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i + 1) for i in range(X_data.shape[1])]  # type: ignore

    p_values = model.pvalues.drop("const", errors="ignore")

    return model, vif_data, p_values


def ols_tuning_process(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[str], str, str, RegressionResultsWrapper]:
    """Performs stepwise OLS tuning based on VIF and P-value."""

    # Use log transformation because below 'L' check confirms there is no linearity in y.
    # sqrt transformation didn't work well.
    y_train = np.log1p(y_train)  # type: ignore
    # y_train = np.sqrt(y_train)  # type: ignore

    # 1. Initial Model Summary (Pre-tuning)
    X_train_sm_initial = sm.add_constant(X_train)
    initial_model = sm.OLS(y_train, X_train_sm_initial).fit()
    print("=" * 70)
    print("1. Initial OLS Model Summary")
    print(initial_model.summary())

    # Is this X and y good for linear regression?
    # Check L.I.N.E. (Linearity, Independence, Normality, Equal Variance)
    print("\n--- Initial Assumption Check ---")
    residuals = initial_model.resid

    # L: Visual check -> If not linear, consider transformations (like log, sqrt) or polynomial terms
    print("-> Linearity: (Visual check recommended)")
    sns.residplot(x=initial_model.fittedvalues, y=residuals, lowess=True)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Fitted Values")
    plt.show()

    # I: Durbin-Watson test for independence -> if DW stat far from 2, indicates autocorrelation
    dw_stat = sm.stats.stattools.durbin_watson(residuals)
    print(f"-> Independence (Durbin-Watson) statistic: {dw_stat:.4f}")

    # N: Shapiro-Wilk test -> if p-value < 0.05, residuals not normal
    shapiro_test = stats.shapiro(residuals)
    print(f"-> Normality (Shapiro-Wilk) p-value: {shapiro_test.pvalue:.4f}")

    # E: Goldfeld-Quandt test -> if p-value < 0.05, heteroscedasticity present
    gq_test = sms.het_goldfeldquandt(residuals, X_train_sm_initial)
    print(f"-> Homoscedasticity (Goldfeld-Quandt) p-value: {gq_test[1]:.4f}")
    print("=" * 70)

    # 2. Tuning Loop
    features = list(X_train.columns)
    tuning_log = []

    print("\n" + "=" * 70)
    print(f"2. Tuning Loop: Feature Selection (VIF > {VIF_THRESHOLD:.1f} OR P-value > {P_VALUE_THRESHOLD:.2f})")

    if "TAX" in features:
        features.remove("TAX")
        tuning_log.append(" [Manual Remove] 'TAX' (Reason: High VIF with RAD).")

    while True:
        current_X = X_train[features]
        if current_X.shape[1] == 0:
            tuning_log.append("Warning: No variables left. Stopping loop.")
            break

        model, vif_data, p_values = check_vif_pvalue_ols(current_X, y_train)

        feature_to_remove = None
        reason = None

        high_vif_features = vif_data[vif_data["VIF"] > VIF_THRESHOLD]

        if not high_vif_features.empty:
            vif_p_values = p_values.loc[high_vif_features["feature"]]
            if vif_p_values.empty or vif_p_values.isna().all():
                feature_to_remove = high_vif_features.sort_values(by="VIF", ascending=False).iloc[0]["feature"]
            else:
                feature_to_remove = vif_p_values.idxmax()
            reason = f"VIF > {VIF_THRESHOLD:.1f} & Max P-value"
        else:
            high_p_value_features = p_values[p_values > P_VALUE_THRESHOLD].index
            if not high_p_value_features.empty:
                high_p_value_idx = np.argmax(p_values[high_p_value_features])
                feature_to_remove = high_p_value_features[high_p_value_idx]
                reason = f"P-value > {P_VALUE_THRESHOLD:.2f}"
            else:
                tuning_log.append("\n--- Tuning Complete ---")
                tuning_log.append(f"All VIFs < {VIF_THRESHOLD:.1f} and all P-values < {P_VALUE_THRESHOLD:.2f}.")
                break

        if feature_to_remove and feature_to_remove in features:
            features.remove(feature_to_remove)
            tuning_log.append(f" [Removed] '{feature_to_remove}' (Reason: {reason}). Left: {features}")
        elif feature_to_remove:
            tuning_log.append(f"Warning: '{feature_to_remove}' not in list. Stopping loop.")
            break

    print("\n".join(tuning_log))
    print("=" * 70)

    # 3. Final Model Summary
    final_X_train_sm = sm.add_constant(X_train[features])
    final_ols_model: RegressionResultsWrapper = sm.OLS(y_train, final_X_train_sm).fit()

    final_summary_text = final_ols_model.summary().as_text()

    print("\n" + "=" * 70)
    print("3. Final OLS Model Summary")
    print(final_summary_text)
    print("=" * 70)

    coef_df = final_ols_model.params.to_frame(name="Coefficient").reset_index().rename(columns={"index": "Feature"})
    equation_parts = []

    intercept = coef_df[coef_df["Feature"] == "const"]["Coefficient"].iloc[0]
    equation_parts.append(f"{TARGET_NAME} = {intercept:.4f}")

    for _, row in coef_df[coef_df["Feature"] != "const"].iterrows():
        feature = row["Feature"]
        coef = row["Coefficient"]
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f" {sign} {abs(coef):.4f} * {feature}")

    final_equation = "".join(equation_parts)
    print(f"\nFinal Equation: {final_equation}")
    print("=" * 70)

    return features, final_equation, final_summary_text, final_ols_model


def train_and_save_sklearn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    final_equation: str,
    final_summary_text: str,
    output_path: str,
):
    """Trains scikit-learn pipeline and saves to output_path."""

    X_final = X_train[selected_features]

    ols_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ols", LinearRegression()),
    ])

    ols_pipeline.fit(X_final, y_train)

    hyperparams = {"VIF_Threshold": VIF_THRESHOLD, "P_Value_Threshold": P_VALUE_THRESHOLD}

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_model_and_metadata(
        model=ols_pipeline,
        params=hyperparams,
        output_path=output_path,
        feature_names=selected_features,
        target_name=TARGET_NAME,
        equation_str=final_equation,
        ols_summary=final_summary_text,
    )

    print(f"\nTraining complete. Model saved to {output_path}")


# --- Click CLI Command ---
@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input Boston Housing CSV file.",
)
@click.option(
    "--output-model",
    "output_model_path",
    required=True,
    type=click.Path(),
    help="Path to save the output joblib model (e.g., models/boston-1.0.1.joblib).",
)
def main_cli(input_path, output_model_path):
    """Train OLS model on Boston Housing data."""

    # Derive metadata path from model path
    metadata_path: str = output_model_path.replace(".joblib", ".json")
    X_train, X_test, y_train, y_test = load_and_prepare_data(input_path)
    final_features, final_equation, final_summary_text, final_ols_model = ols_tuning_process(X_train, y_train)
    if final_features:
        train_and_save_sklearn_model(
            X_train,
            y_train,
            final_features,
            final_equation,
            final_summary_text,
            output_model_path,
        )
    else:
        print("No final features selected. Skipping model save.")


if __name__ == "__main__":
    main_cli()
