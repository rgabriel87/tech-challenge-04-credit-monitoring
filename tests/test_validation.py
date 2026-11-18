"""Testes automatizados do Data Contract de Credit Scoring."""

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.validate import (
    create_invalid_batch,
    validate_credit_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference.csv"


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Carrega o dataset de referência usado nos testes."""
    return pd.read_csv(REFERENCE_PATH)


def test_reference_dataset_is_valid(
    reference_data: pd.DataFrame,
) -> None:
    """O dataset de referência deve atender ao Data Contract."""

    validated = validate_credit_data(
        reference_data
    )

    assert len(validated) == 10_000
    assert validated.isnull().sum().sum() == 0


def test_invalid_batch_is_blocked(
    reference_data: pd.DataFrame,
) -> None:
    """Um lote contendo violações deve ser bloqueado."""

    invalid_batch = create_invalid_batch(
        reference_data
    )

    with pytest.raises(SchemaErrors):
        validate_credit_data(
            invalid_batch
        )


def test_age_below_18_is_blocked(
    reference_data: pd.DataFrame,
) -> None:
    """Clientes menores de 18 anos devem ser rejeitados."""

    invalid = reference_data.head(10).copy()

    invalid.loc[
        invalid.index[0],
        "age",
    ] = 15

    with pytest.raises(SchemaErrors):
        validate_credit_data(
            invalid
        )


def test_null_income_is_blocked(
    reference_data: pd.DataFrame,
) -> None:
    """Renda nula deve ser rejeitada pelo contrato."""

    invalid = reference_data.head(10).copy()

    invalid.loc[
        invalid.index[0],
        "income",
    ] = None

    with pytest.raises(SchemaErrors):
        validate_credit_data(
            invalid
        )


def test_invalid_credit_score_is_blocked(
    reference_data: pd.DataFrame,
) -> None:
    """Credit score fora do intervalo permitido deve ser rejeitado."""

    invalid = reference_data.head(10).copy()

    invalid.loc[
        invalid.index[0],
        "credit_score",
    ] = 950

    with pytest.raises(SchemaErrors):
        validate_credit_data(
            invalid
        )


def test_duplicate_customer_id_is_blocked(
    reference_data: pd.DataFrame,
) -> None:
    """IDs duplicados devem ser rejeitados."""

    invalid = reference_data.head(10).copy()

    invalid.loc[
        invalid.index[0],
        "customer_id",
    ] = invalid.loc[
        invalid.index[1],
        "customer_id",
    ]

    with pytest.raises(SchemaErrors):
        validate_credit_data(
            invalid
        )