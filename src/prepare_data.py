"""Geração do dataset de referência para Credit Scoring."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_PATH = DATA_DIR / "reference.csv"

RANDOM_SEED = 42
N_SAMPLES = 10_000


def create_reference_dataset() -> pd.DataFrame:
    """Cria um dataset sintético e reprodutível de risco de crédito."""
    rng = np.random.default_rng(RANDOM_SEED)

    age = rng.integers(18, 76, N_SAMPLES)
    income = np.clip(
        rng.normal(loc=5500, scale=2500, size=N_SAMPLES),
        1200,
        25000,
    ).round(2)

    debt_ratio = np.clip(
        rng.beta(2, 5, N_SAMPLES),
        0,
        1,
    ).round(4)

    credit_score = np.clip(
        rng.normal(loc=650, scale=90, size=N_SAMPLES),
        300,
        850,
    ).round().astype(int)

    employment_years = np.clip(
        rng.exponential(scale=6, size=N_SAMPLES),
        0,
        40,
    ).round(1)

    late_payments = np.clip(
        rng.poisson(lam=1.5, size=N_SAMPLES),
        0,
        10,
    )

    loan_amount = np.clip(
        rng.normal(loc=18_000, scale=10_000, size=N_SAMPLES),
        1_000,
        80_000,
    ).round(2)

    # Quanto maior o valor, maior o risco de inadimplência.
    risk_logit = (
        -2.0
        + 3.2 * debt_ratio
        - 0.006 * (credit_score - 600)
        + 0.28 * late_payments
        + 0.000025 * loan_amount
        - 0.00008 * income
        - 0.025 * employment_years
    )

    probability_default = 1 / (1 + np.exp(-risk_logit))
    default = rng.binomial(1, probability_default)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, N_SAMPLES + 1),
            "age": age,
            "income": income,
            "debt_ratio": debt_ratio,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "late_payments": late_payments,
            "loan_amount": loan_amount,
            "default": default,
        }
    )

    return df


def main() -> None:
    """Gera e salva o dataset de referência."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = create_reference_dataset()
    df.to_csv(REFERENCE_PATH, index=False)

    print("Dataset de referência criado com sucesso.")
    print(f"Arquivo: {REFERENCE_PATH}")
    print(f"Registros: {len(df)}")
    print(f"Colunas: {len(df.columns)}")
    print()
    print("Distribuição do target:")
    print(df["default"].value_counts())
    print()
    print("Taxa de inadimplência:")
    print(f"{df['default'].mean():.2%}")


if __name__ == "__main__":
    main()