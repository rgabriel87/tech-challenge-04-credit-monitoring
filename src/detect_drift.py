"""Detecção estatística de Data Drift e geração de relatório com Evidently."""

from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
HTML_REPORT_PATH = REPORTS_DIR / "drift_report.html"
STATISTICS_PATH = REPORTS_DIR / "drift_statistics.csv"


FEATURES = [
    "age",
    "income",
    "debt_ratio",
    "credit_score",
    "employment_years",
    "late_payments",
    "loan_amount",
]


def calculate_ks_statistics(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula o teste Kolmogorov-Smirnov para cada feature numérica."""

    results = []

    for feature in FEATURES:

        statistic, p_value = ks_2samp(
            reference[feature].dropna(),
            production[feature].dropna(),
        )

        drift_detected = p_value < 0.05

        results.append(
            {
                "feature": feature,
                "ks_statistic": statistic,
                "p_value": p_value,
                "drift_detected": drift_detected,
                "reference_mean": reference[feature].mean(),
                "production_mean": production[feature].mean(),
            }
        )

    return pd.DataFrame(results)


def generate_evidently_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> None:
    """Gera relatório HTML de Data Drift utilizando Evidently."""

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference[FEATURES],
        current_data=production[FEATURES],
    )

    result.save_html(
        str(HTML_REPORT_PATH)
    )


def main() -> None:
    """Executa a análise completa de Data Drift."""

    print("=== DETECÇÃO DE DATA DRIFT ===")
    print()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = pd.read_csv(
        REFERENCE_PATH
    )

    production = pd.read_csv(
        PRODUCTION_PATH
    )

    print(
        f"Reference Dataset: {len(reference)} registros"
    )

    print(
        f"Production Dataset: {len(production)} registros"
    )

    print()
    print("Executando testes estatísticos KS...")
    print()

    statistics = calculate_ks_statistics(
        reference,
        production,
    )

    statistics.to_csv(
        STATISTICS_PATH,
        index=False,
    )

    print("=== RESULTADOS DO TESTE KS ===")
    print()

    for _, row in statistics.iterrows():

        status = (
            "DRIFT"
            if row["drift_detected"]
            else "OK"
        )

        print(
            f"{row['feature']:<20} "
            f"KS={row['ks_statistic']:.4f} "
            f"p={row['p_value']:.6f} "
            f"[{status}]"
        )

    drifted_features = statistics[
        statistics["drift_detected"]
    ]["feature"].tolist()

    print()
    print(
        f"Features com drift: "
        f"{len(drifted_features)}/{len(FEATURES)}"
    )

    if drifted_features:
        print(
            "Detectadas: "
            + ", ".join(drifted_features)
        )

    print()
    print(
        f"Estatísticas salvas em: "
        f"{STATISTICS_PATH}"
    )

    print()
    print("Gerando relatório Evidently...")

    generate_evidently_report(
        reference,
        production,
    )

    print()
    print("Relatório Evidently gerado com sucesso.")
    print(
        f"Arquivo: {HTML_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()