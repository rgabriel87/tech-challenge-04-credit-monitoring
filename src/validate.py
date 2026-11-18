"""Contrato e validação de qualidade dos dados de Credit Scoring."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_PATH = DATA_DIR / "reference.csv"
INVALID_PATH = DATA_DIR / "invalid_batch.csv"


credit_schema = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(
            int,
            pa.Check.greater_than(0),
            unique=True,
            nullable=False,
        ),
        "age": pa.Column(
            int,
            [
                pa.Check.greater_than_or_equal_to(18),
                pa.Check.less_than_or_equal_to(100),
            ],
            nullable=False,
        ),
        "income": pa.Column(
            float,
            pa.Check.greater_than(0),
            nullable=False,
        ),
        "debt_ratio": pa.Column(
            float,
            [
                pa.Check.greater_than_or_equal_to(0),
                pa.Check.less_than_or_equal_to(1),
            ],
            nullable=False,
        ),
        "credit_score": pa.Column(
            int,
            [
                pa.Check.greater_than_or_equal_to(300),
                pa.Check.less_than_or_equal_to(850),
            ],
            nullable=False,
        ),
        "employment_years": pa.Column(
            float,
            pa.Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "late_payments": pa.Column(
            int,
            pa.Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "loan_amount": pa.Column(
            float,
            pa.Check.greater_than(0),
            nullable=False,
        ),
        "default": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
        ),
    },
    strict=True,
    coerce=True,
)


def validate_credit_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Valida um DataFrame conforme o contrato de dados."""

    return credit_schema.validate(
        df,
        lazy=True,
    )


def create_invalid_batch(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Cria propositalmente um lote inválido para testar o contrato."""

    invalid = df.head(20).copy()

    # Violações intencionais do Data Contract.
    invalid.loc[0, "age"] = 15
    invalid.loc[1, "income"] = None
    invalid.loc[2, "credit_score"] = 950
    invalid.loc[3, "debt_ratio"] = 1.5

    # Cria customer_id duplicado.
    invalid.loc[4, "customer_id"] = invalid.loc[
        5,
        "customer_id",
    ]

    return invalid


def main() -> None:
    """Demonstra aprovação do dataset válido e bloqueio do lote inválido."""

    print("=== DATA CONTRACT - CREDIT SCORING ===")
    print()

    reference = pd.read_csv(
        REFERENCE_PATH
    )

    print(
        "[1] Validando Dataset de Referência..."
    )

    try:

        validate_credit_data(
            reference
        )

        print(
            f"APROVADO: "
            f"{len(reference)} registros válidos."
        )

    except pa.errors.SchemaErrors as exc:

        print(
            "ERRO: Dataset de referência reprovado."
        )

        print(
            exc.failure_cases
        )

        raise

    print()
    print(
        "[2] Criando lote propositalmente inválido..."
    )

    invalid_batch = create_invalid_batch(
        reference
    )

    invalid_batch.to_csv(
        INVALID_PATH,
        index=False,
    )

    print(
        f"Lote inválido salvo em: "
        f"{INVALID_PATH}"
    )

    print()
    print(
        "[3] Executando o Data Contract "
        "sobre o lote inválido..."
    )

    try:

        validate_credit_data(
            invalid_batch
        )

        print(
            "ATENÇÃO: lote inválido foi aprovado."
        )

    except pa.errors.SchemaErrors as exc:

        print()
        print(
            "BLOQUEADO PELO DATA CONTRACT."
        )

        print(
            "A ingestão NÃO deve continuar."
        )

        print()
        print(
            "Anomalias encontradas:"
        )

        columns = [
            "column",
            "failure_case",
            "index",
            "check",
        ]

        available = [
            column
            for column in columns
            if column in exc.failure_cases.columns
        ]

        print(
            exc.failure_cases[
                available
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()