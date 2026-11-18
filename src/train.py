"""Treinamento do modelo baseline de Credit Scoring com MLflow."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.validate import validate_credit_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "reference.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_model.joblib"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

FEATURES = [
    "age",
    "income",
    "debt_ratio",
    "credit_score",
    "employment_years",
    "late_payments",
    "loan_amount",
]

TARGET = "default"

RANDOM_SEED = 42


def load_data() -> pd.DataFrame:
    """Carrega e valida o dataset de referência."""
    df = pd.read_csv(DATA_PATH)

    validated_df = validate_credit_data(df)

    return validated_df


def train_model():
    """Treina o modelo baseline e registra o experimento no MLflow."""

    print("Carregando e validando os dados...")

    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # MLflow utilizando SQLite como backend local.
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-scoring-baseline")

    print("Treinando modelo...")

    with mlflow.start_run() as run:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
        }

        parameters = {
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "test_size": 0.20,
            "random_state": RANDOM_SEED,
        }

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        # O modelo foi criado localmente por este projeto.
        # O tipo abaixo é utilizado internamente pelo RandomForest.
        mlflow.sklearn.log_model(
            model,
            name="credit_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
            ],
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            MODEL_PATH,
        )

        print()
        print("=== TREINAMENTO BASELINE ===")
        print()

        print(f"Registros de treino: {len(X_train)}")
        print(f"Registros de teste: {len(X_test)}")

        print()
        print("=== MÉTRICAS ===")

        for metric_name, metric_value in metrics.items():
            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )

        print()
        print(f"Modelo salvo em: {MODEL_PATH}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow Tracking URI: {tracking_uri}")

    return model, metrics


if __name__ == "__main__":
    train_model()