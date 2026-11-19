"""Ingeniería de características (V1.1.0).

Primer componente del flujo de modelos: toma la base cruda, la divide en
entrenamiento y evaluación y la convierte en la tabla numérica con la que se
entrenan los modelos.

Todo lo que "aprende" de los datos (medianas, categorías raras, escalas) se
ajusta solo con el conjunto de entrenamiento y después se aplica al de
evaluación. Así no se filtra información de evaluación al entrenamiento.

Las decisiones vienen de comprension_eda.ipynb (secciones 5.2 y 5.4). Los
números entre paréntesis, como "supuesto 3", se refieren a la tabla de 5.4.

Uso:
    python src/ft_engineering.py
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.encoding import RareLabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Base_de_datos.csv"

TARGET = "Pago_atiempo"  # 1 = pagó a tiempo, 0 = no pagó a tiempo
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Umbrales de limpieza. Son criterio propio (supuestos 3, 4, 5 y 10).
AGE_MIN, AGE_MAX = 18, 100  # las edades de 121 a 123 parecen un código
MONEY_MAX = 100_000_000  # salarios y otros préstamos por encima son errores
GROUP_COL = "tipo_laboral"  # grupo para imputar con la mediana (supuesto 11)
INCOME_TREND_CODES = {"Decreciente": 0, "Estable": 1, "Creciente": 2}  # supuesto 8
INCOME_TREND_MISSING_CODE = 1  # los nulos reciben el valor del medio
RARE_CREDIT_TYPE_TOL = 0.02  # tipo_credito con menos de 2% pasa a "otros"

# Columnas que nunca entran al modelo.
# - fecha_prestamo: solo sirve para dividir en el tiempo; como variable
#   mezclaría el modelo con la fecha de la solicitud.
# - saldo_mora_codeudor: casi constante y con 590 nulos (supuesto 9).
ALWAYS_DROPPED = ["fecha_prestamo", "saldo_mora_codeudor"]
# puntaje se excluye por probable fuga de información (supuesto 2).
SCORE_COL = "puntaje"

# Columnas con asimetría fuerte: se transforman con log(1 + x).
LOG_COLUMNS = [
    "capital_prestado",
    "cuota_pactada",
    "salario_cliente",
    "total_otros_prestamos",
    "promedio_ingresos_datacredito",
]
RATIO_COLUMNS = [
    "cuota_sobre_ingreso_datacredito",
    "capital_sobre_salario",
    "otros_prestamos_sobre_salario",
]

# Columnas a las que se les aplica la mediana por grupo.
GROUP_IMPUTED_COLUMNS = [
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "puntaje_datacredito",
]

FLAG_COLUMNS = [
    "edad_corregida",
    "salario_corregido",
    "otros_prestamos_corregido",
    "puntaje_datacredito_corregido",
    "sin_historial_ingresos",
    "saldo_nulo",
    "saldo_principal_nulo",
    "tiene_mora",
]


class DataCleaner(BaseEstimator, TransformerMixin):
    """Corrige tipos y valores imposibles, y marca dónde se corrigió.

    No aprende nada de los datos, así que fit() no hace falta. Los valores
    imposibles pasan a nulo y se agrega un indicador 0/1 por cada variable
    corregida. Los nulos se rellenan en los pasos siguientes.
    """

    def __init__(self, include_score=False):
        self.include_score = include_score

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # tipo_credito es un código, no una cantidad.
        X["tipo_credito"] = X["tipo_credito"].astype("int64").astype(str)

        # Valores imposibles o con aspecto de código pasan a nulo.
        age = X["edad_cliente"].where(X["edad_cliente"].between(AGE_MIN, AGE_MAX))
        salary = X["salario_cliente"].where(X["salario_cliente"].between(1, MONEY_MAX))
        other_loans = X["total_otros_prestamos"].where(X["total_otros_prestamos"] <= MONEY_MAX)
        credit_score = X["puntaje_datacredito"].where(X["puntaje_datacredito"] > 0)
        income = X["promedio_ingresos_datacredito"].where(X["promedio_ingresos_datacredito"] > 0)

        # Los 58 valores numéricos de tendencia_ingresos no son categorías (supuesto 6).
        trend = X["tendencia_ingresos"].map(INCOME_TREND_CODES)  # lo que no es categoría queda nulo

        # Indicadores. Se calculan antes de imputar para conservar la señal.
        X["edad_corregida"] = age.isna().astype(int)
        X["salario_corregido"] = salary.isna().astype(int)
        X["otros_prestamos_corregido"] = other_loans.isna().astype(int)
        X["puntaje_datacredito_corregido"] = credit_score.isna().astype(int)
        # Los nulos de ingresos se dejan sin imputar y se marcan (supuesto 7).
        X["sin_historial_ingresos"] = income.isna().astype(int)
        X["saldo_nulo"] = (X["saldo_mora"].isna() | X["saldo_total"].isna()).astype(int)
        X["saldo_principal_nulo"] = X["saldo_principal"].isna().astype(int)
        X["tiene_mora"] = (X["saldo_mora"] > 0).astype(int)

        X["edad_cliente"] = age
        X["salario_cliente"] = salary
        X["total_otros_prestamos"] = other_loans
        X["puntaje_datacredito"] = credit_score
        X["promedio_ingresos_datacredito"] = income
        # tendencia_ingresos es ordinal. El valor del medio para los nulos es neutro porque
        # sin_historial_ingresos ya lleva la información de que falta.
        X["tendencia_ingresos"] = trend.fillna(INCOME_TREND_MISSING_CODE).astype(int)

        to_drop = list(ALWAYS_DROPPED)
        if not self.include_score:
            to_drop.append(SCORE_COL)
        return X.drop(columns=to_drop, errors="ignore")


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Rellena nulos con la mediana de cada grupo (por defecto, tipo_laboral).

    Las medianas se calculan en fit(), solo con los datos de entrenamiento.
    Si aparece un grupo que no se vio al entrenar, se usa la mediana global.
    """

    def __init__(self, columns, group_col=GROUP_COL):
        self.columns = columns
        self.group_col = group_col

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group_col)[list(self.columns)].median()
        self.global_medians_ = X[list(self.columns)].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            by_group = X[self.group_col].map(self.group_medians_[col]).astype(float)
            fill = by_group.fillna(self.global_medians_[col])
            X[col] = X[col].fillna(fill)
        return X


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Crea los cocientes (sección 5.3 del EDA) y aplica log(1 + x).

    Las columnas transformadas se renombran con el prefijo log_. Los cocientes
    se calculan antes del logaritmo, con los valores ya imputados.
    """

    def __init__(self, add_ratios=True):
        self.add_ratios = add_ratios

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        log_columns = list(LOG_COLUMNS)

        if self.add_ratios:
            income = X["promedio_ingresos_datacredito"]  # puede ser nulo
            X["cuota_sobre_ingreso_datacredito"] = X["cuota_pactada"] / income
            X["capital_sobre_salario"] = X["capital_prestado"] / X["salario_cliente"]
            X["otros_prestamos_sobre_salario"] = X["total_otros_prestamos"] / X["salario_cliente"]
            log_columns += RATIO_COLUMNS

        for col in log_columns:
            X[f"log_{col}"] = np.log1p(X.pop(col))
        return X


def load_data(path=DATA_PATH):
    """Lee la base y deja los tipos tal como los usa el EDA (sección 1.9)."""
    return pd.read_csv(path, parse_dates=["fecha_prestamo"])


def split_data(df, mode="stratified", test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Divide en entrenamiento y evaluación. Devuelve X_train, X_test, y_train, y_test.

    mode="stratified": partición aleatoria que conserva la proporción de
        impagos (4.7%) en los dos conjuntos.
    mode="temporal": los préstamos más antiguos entrenan y los más recientes
        evalúan. Se acerca más al uso real, donde el modelo predice el futuro.
    """
    if mode == "stratified":
        X = df.drop(columns=[TARGET])
        return train_test_split(
            X, df[TARGET], test_size=test_size, stratify=df[TARGET], random_state=random_state
        )
    if mode == "temporal":
        ordered = df.sort_values("fecha_prestamo")
        cut = int(len(ordered) * (1 - test_size))
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
        return (
            train.drop(columns=[TARGET]),
            test.drop(columns=[TARGET]),
            train[TARGET],
            test[TARGET],
        )
    raise ValueError("mode debe ser 'stratified' o 'temporal'")


def build_preprocessor(include_score=False, add_ratios=True):
    """Arma el pipeline completo, sin ajustar.

    Pasos: limpieza -> mediana por tipo laboral -> atributos derivados ->
    codificación y escalado por tipo de variable.
    """
    numeric = [
        "plazo_meses",
        "edad_cliente",
        "puntaje_datacredito",
        "cant_creditosvigentes",
        "huella_consulta",
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
    ] + [f"log_{c}" for c in LOG_COLUMNS]
    if add_ratios:
        numeric += [f"log_{c}" for c in RATIO_COLUMNS]
    if include_score:
        numeric.append(SCORE_COL)

    # Los nulos que quedan (ingresos y saldos) se rellenan con la mediana del
    # entrenamiento. Es seguro porque cada uno tiene su indicador 0/1.
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    # tipo_credito: los códigos con pocas filas (6, 7, 10 y 68) pasan a "otros".
    credit_type_pipe = Pipeline(
        [
            (
                "rare",
                RareLabelEncoder(
                    tol=RARE_CREDIT_TYPE_TOL, n_categories=2, replace_with="otros"
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    columns = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("flags", "passthrough", FLAG_COLUMNS),
            ("tipo_laboral", OneHotEncoder(drop="if_binary", sparse_output=False), [GROUP_COL]),
            ("tipo_credito", credit_type_pipe, ["tipo_credito"]),
            ("tendencia", "passthrough", ["tendencia_ingresos"]),  # ya viene codificada 0, 1, 2
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline(
        [
            ("clean", DataCleaner(include_score=include_score)),
            ("impute_by_group", GroupMedianImputer(GROUP_IMPUTED_COLUMNS)),
            ("derived", DerivedFeatures(add_ratios=add_ratios)),
            ("columns", columns),
        ]
    )


@dataclass
class ModelData:
    """Todo lo que necesita el entrenamiento.

    X_train y X_test ya están transformados. Las versiones *_raw y el
    preprocesador sin ajustar (preprocessor_template) sirven para validación
    cruzada: en cada partición hay que ajustar el pipeline de nuevo.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    preprocessor: Pipeline  # ajustado con el entrenamiento
    preprocessor_template: Pipeline  # sin ajustar


def build_datasets(
    include_score=False,
    add_ratios=True,
    split_mode="stratified",
    path=DATA_PATH,
):
    """Carga, divide y transforma. Devuelve un ModelData."""
    df = load_data(path)
    X_train_raw, X_test_raw, y_train, y_test = split_data(df, mode=split_mode)

    template = build_preprocessor(include_score=include_score, add_ratios=add_ratios)

    preprocessor = clone(template)
    X_train = preprocessor.fit_transform(X_train_raw)  # se ajusta solo con entrenamiento
    X_test = preprocessor.transform(X_test_raw)

    return ModelData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        preprocessor=preprocessor,
        preprocessor_template=template,
    )


def main():
    data = build_datasets()
    print("Entrenamiento:", data.X_train.shape, "| Evaluación:", data.X_test.shape)
    print(
        "Impagos (Pago_atiempo = 0): "
        f"{(data.y_train == 0).mean():.1%} en entrenamiento, "
        f"{(data.y_test == 0).mean():.1%} en evaluación"
    )
    print("Nulos restantes:", int(data.X_train.isna().sum().sum() + data.X_test.isna().sum().sum()))
    print("Variables de entrada al modelo:")
    for name in data.X_train.columns:
        print(" -", name)


if __name__ == "__main__":
    main()
