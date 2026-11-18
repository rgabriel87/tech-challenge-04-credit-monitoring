"""Simulação de Data Drift em dados de produção de Credit Scoring."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train import FEATURES
from src.validate import validate_credit_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_model.joblib"

RANDOM_SEED = 123


def simulate_production_data(
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Cria dados de produção com mudanças intencionais de distribuição."""

    rng = np.random.default_rng(RANDOM_SEED)

    # Amostra que representa um período posterior em produção.
    production = reference.sample(
        n=5000,
        random_state=RANDOM_SEED,
    ).copy()

    # ---------------------------------------------------------
    # DRIFT 1 - RENDA
    # Simula perda de poder aquisitivo / deterioração econômica.
    # ---------------------------------------------------------
    production["income"] = (
        production["income"] * 0.70
    ).clip(lower=500).round(2)

    # ---------------------------------------------------------
    # DRIFT 2 - ENDIVIDAMENTO
    # Clientes passam a comprometer uma parcela maior da renda.
    # ---------------------------------------------------------
    production["debt_ratio"] = (
        production["debt_ratio"] + 0.20
    ).clip(0, 1).round(4)

    # ---------------------------------------------------------
    # DRIFT 3 - ATRASOS
    # Aumenta a quantidade de pagamentos atrasados.
    # ---------------------------------------------------------
    additional_late_payments = rng.poisson(
        lam=2.0,
        size=len(production),
    )

    production["late_payments"] = (
        production["late_payments"]
        + additional_late_payments
    ).clip(0, 10)

    # Novos IDs simulam novos clientes chegando em produção.
    production["customer_id"] = np.arange(
        100_001,
        100_001 + len(production),
    )

    return production.reset_index(drop=True)


def generate_predictions(
    production: pd.DataFrame,
) -> pd.DataFrame:
    """Executa o modelo baseline sobre os dados degradados."""

    model = joblib.load(MODEL_PATH)

    production = production.copy()

    production["prediction"] = model.predict(
        production[FEATURES]
    )

    production["default_probability"] = model.predict_proba(
        production[FEATURES]
    )[:, 1]

    return production


def main() -> None:
    """Gera o dataset de produção contendo drift controlado."""

    print("=== SIMULAÇÃO DE DATA DRIFT ===")
    print()

    reference = pd.read_csv(REFERENCE_PATH)

    production = simulate_production_data(reference)

    # Valida apenas as colunas previstas pelo contrato.
    validate_credit_data(
        production[reference.columns]
    )

    production = generate_predictions(production)

    production.to_csv(
        PRODUCTION_PATH,
        index=False,
    )

    print(f"Dataset de referência: {len(reference)} registros")
    print(f"Dataset de produção: {len(production)} registros")

    print()
    print("=== COMPARAÇÃO DAS DISTRIBUIÇÕES ===")
    print()

    variables = [
        "income",
        "debt_ratio",
        "late_payments",
    ]

    for variable in variables:
        reference_mean = reference[variable].mean()
        production_mean = production[variable].mean()

        change = (
            (production_mean - reference_mean)
            / reference_mean
            * 100
        )

        print(variable)
        print(f"  Referência: {reference_mean:.4f}")
        print(f"  Produção:   {production_mean:.4f}")
        print(f"  Mudança:    {change:+.2f}%")
        print()

    print("=== COMPORTAMENTO DO MODELO ===")
    print()

    print(
        "Taxa prevista de inadimplência: "
        f"{production['prediction'].mean():.2%}"
    )

    print(
        "Probabilidade média de inadimplência: "
        f"{production['default_probability'].mean():.2%}"
    )

    print()
    print(f"Dataset salvo em: {PRODUCTION_PATH}")


if __name__ == "__main__":
    main()