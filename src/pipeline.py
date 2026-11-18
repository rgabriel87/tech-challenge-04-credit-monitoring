"""Pipeline monitorado de Credit Scoring.

Executa as etapas principais do fluxo e registra métricas operacionais
de observabilidade em arquivo de log.
"""

from pathlib import Path
import logging
import time

import pandas as pd

from src.detect_drift import (
    FEATURES,
    calculate_ks_statistics,
    generate_evidently_report,
)
from src.simulate_drift import (
    generate_predictions,
    simulate_production_data,
)
from src.validate import validate_credit_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production.csv"

LOGS_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOGS_DIR / "pipeline.log"

REPORTS_DIR = PROJECT_ROOT / "reports"
STATISTICS_PATH = REPORTS_DIR / "drift_statistics.csv"

DRIFT_ALERT_THRESHOLD = 1


def configure_logging() -> logging.Logger:
    """Configura logging em arquivo e terminal."""

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("credit_monitoring_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def run_pipeline() -> None:
    """Executa o pipeline monitorado."""

    logger = configure_logging()

    start_time = time.perf_counter()

    pipeline_status = "SUCCESS"
    alerts = 0

    logger.info("=" * 60)
    logger.info("PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        # -----------------------------------------------------
        # ETAPA 1 - CARREGAMENTO
        # -----------------------------------------------------

        logger.info(
            "STEP=LOAD_REFERENCE STATUS=STARTED"
        )

        reference = pd.read_csv(
            REFERENCE_PATH
        )

        logger.info(
            "STEP=LOAD_REFERENCE STATUS=SUCCESS "
            "RECORDS=%d",
            len(reference),
        )

        # -----------------------------------------------------
        # ETAPA 2 - DATA CONTRACT
        # -----------------------------------------------------

        logger.info(
            "STEP=DATA_VALIDATION STATUS=STARTED"
        )

        validate_credit_data(reference)

        logger.info(
            "STEP=DATA_VALIDATION STATUS=SUCCESS "
            "RECORDS=%d",
            len(reference),
        )

        # -----------------------------------------------------
        # ETAPA 3 - SIMULAÇÃO DE PRODUÇÃO
        # -----------------------------------------------------

        logger.info(
            "STEP=PRODUCTION_SIMULATION STATUS=STARTED"
        )

        production = simulate_production_data(
            reference
        )

        validate_credit_data(
            production[reference.columns]
        )

        production = generate_predictions(
            production
        )

        production.to_csv(
            PRODUCTION_PATH,
            index=False,
        )

        logger.info(
            "STEP=PRODUCTION_SIMULATION STATUS=SUCCESS "
            "RECORDS=%d",
            len(production),
        )

        # -----------------------------------------------------
        # ETAPA 4 - DETECÇÃO DE DRIFT
        # -----------------------------------------------------

        logger.info(
            "STEP=DRIFT_DETECTION STATUS=STARTED"
        )

        statistics = calculate_ks_statistics(
            reference,
            production,
        )

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        statistics.to_csv(
            STATISTICS_PATH,
            index=False,
        )

        drifted_features = statistics[
            statistics["drift_detected"]
        ]["feature"].tolist()

        drift_count = len(drifted_features)

        logger.info(
            "STEP=DRIFT_DETECTION STATUS=SUCCESS "
            "DRIFTED_FEATURES=%d TOTAL_FEATURES=%d",
            drift_count,
            len(FEATURES),
        )

        if drift_count >= DRIFT_ALERT_THRESHOLD:

            alerts += 1

            logger.warning(
                "ALERT=DATA_DRIFT "
                "DRIFTED_FEATURES=%s",
                ",".join(drifted_features),
            )

        # -----------------------------------------------------
        # ETAPA 5 - EVIDENTLY
        # -----------------------------------------------------

        logger.info(
            "STEP=EVIDENTLY_REPORT STATUS=STARTED"
        )

        generate_evidently_report(
            reference,
            production,
        )

        logger.info(
            "STEP=EVIDENTLY_REPORT STATUS=SUCCESS"
        )

        # -----------------------------------------------------
        # MÉTRICAS DO MODELO EM PRODUÇÃO
        # -----------------------------------------------------

        predicted_default_rate = (
            production["prediction"].mean()
        )

        average_default_probability = (
            production["default_probability"].mean()
        )

        logger.info(
            "METRIC=PREDICTED_DEFAULT_RATE VALUE=%.4f",
            predicted_default_rate,
        )

        logger.info(
            "METRIC=AVERAGE_DEFAULT_PROBABILITY VALUE=%.4f",
            average_default_probability,
        )

    except Exception:

        pipeline_status = "FAILED"
        alerts += 1

        logger.exception(
            "PIPELINE FAILURE"
        )

        raise

    finally:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.info(
            "METRIC=PIPELINE_DURATION_SECONDS VALUE=%.4f",
            duration,
        )

        logger.info(
            "METRIC=ALERT_COUNT VALUE=%d",
            alerts,
        )

        logger.info(
            "PIPELINE STATUS=%s",
            pipeline_status,
        )

        logger.info("=" * 60)

        print()
        print("=== RESUMO DE OBSERVABILIDADE ===")
        print()
        print(f"Status do pipeline: {pipeline_status}")
        print(f"Duração: {duration:.2f} segundos")
        print(f"Alertas: {alerts}")
        print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    run_pipeline()