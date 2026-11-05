"""
data_utils.py
-------------
A reusable utility module for exploratory data analysis (EDA),
preprocessing, feature selection, and model training.
"""

from operator import lt
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def find_bestmodel(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 3. Feature Scaling (Crucial for Ridge & Lasso)
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 4. Train Models with Cross-Validation for Hyperparameter Tuning
    # ---------------------------------------------------------
    alphas = np.logspace(-4, 4, 100)

    models = {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=alphas, cv=5),
        "Lasso Regression": LassoCV(alphas=alphas, cv=5, random_state=42),
        "ElasticNet Regression": ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        best_alpha = getattr(model, 'alpha_', 'N/A')
        
        results.append({
            "Model": name,
            "Best Alpha": best_alpha if best_alpha == 'N/A' else round(best_alpha, 4),
            "Train RMSE": round(train_rmse, 2),
            "Test RMSE": round(test_rmse, 2),
            "Test MAE": round(test_mae, 2),
            "Test R²": round(test_r2, 4)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values(by="Test R²", ascending=False)

    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string(index=False))

    # Highlight the best model
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n🏆 **Best Suited Model:** {best_model_name}")

    # ---------------------------------------------------------
    # 5. Feature Importance Analysis (Coefficients)
    # ---------------------------------------------------------
    best_model = trained_models[best_model_name]
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print(f"\n--- Feature Coefficients ({best_model_name}) ---")
    print(coef_df.to_string(index=False))

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='viridis')
    plt.title(f'Feature Importance ({best_model_name})')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

def data_exploration(data: pd.DataFrame) -> None:
    """Print key descriptive statistics and metadata of the dataset."""
    print("=" * 40)
    print("DATA EXPLORATION")
    print("=" * 40)
    print(f"Shape of dataset: {data.shape[0]} rows, {data.shape[1]} columns\n")
    print("Columns:", list(data.columns))
    print("\nData Types:\n", data.dtypes)
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nStatistical Summary:\n", data.describe(include="all"))
    print("=" * 40)


def get_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix for numeric columns only."""
    return data.corr(numeric_only=True)


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "coolwarm",
) -> None:
    """Plot a heatmap for a given correlation matrix."""
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar=True,
        square=True,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def handle_missing_values(
    data: pd.DataFrame, numeric_only: bool = True
) -> pd.DataFrame:
    """
    Fill missing values. By default, fills numeric columns with their mean
    and non-numeric columns with their mode.
    """
    df = data.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    if not numeric_only:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
    return df


def normalize_data(
    data: pd.DataFrame, columns: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Scale numeric features to a range of [0, 1] using MinMaxScaler.
    If no columns specified, scales all numeric columns.
    """
    df = data.copy()
    target_cols = (
        columns if columns is not None else df.select_dtypes(include=[np.number]).columns
    )

    scaler = MinMaxScaler()
    df[target_cols] = scaler.fit_transform(df[target_cols])
    return df


def feature_selection(
    data: pd.DataFrame, target_variable: str, k: int = 5
) -> List[str]:
    """Select the top k features using univariate linear regression tests (f_regression)."""
    X = data.drop(columns=[target_variable]).select_dtypes(include=[np.number])
    y = data[target_variable]

    k_features = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k_features)
    selector.fit(X, y)

    selected_columns = X.columns[selector.get_support()].tolist()
    return selected_columns

def plot_actual_vs_predicted(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    figsize: Tuple[int, int] = (6, 4),
) -> None:
    """Plot actual vs predicted values for regression models."""
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

def linear_regression_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[LinearRegression, float, float, Union[pd.Series, np.ndarray], np.ndarray]:
    """
    Split data, train an OLS Linear Regression model, evaluate metrics,
    and return the model, MSE, R-squared, y_test, and y_pred.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    return model, mse, r2, y_test, y_pred




def cross_validation(
    model: LinearRegression,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    scoring: str = "r2",
) -> np.ndarray:
    """Perform k-fold cross-validation and print score distribution."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"{cv}-Fold Cross-Validation ({scoring}): {scores}")
    print(f"Average CV score: {scores.mean():.4f} (± {scores.std():.4f})")
    return scores

def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Filter rows that have values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    df_clean = df.copy()
    for col in columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Retain only data within the bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def plot_boxplots(df: pd.DataFrame, columns: list, figsize: Tuple[int, int] = (10, 4)) -> None:
    """Plot boxplots for specified columns to visualize outliers."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=df[columns], orient='h', palette='Set2')
    plt.title("Box Plot for Outlier Detection (Dots = Outliers)")
    plt.xlabel("Value")
    plt.tight_layout()
    plt.show()