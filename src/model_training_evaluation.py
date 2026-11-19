"""Entrenamiento y evaluación de modelos supervisados (V1.1.1).

Segundo componente del flujo: toma los conjuntos que prepara ft_engineering.py,
entrena varios modelos, los compara y elige el mejor.

Cómo se decide:
- El modelo se elige con validación cruzada sobre el entrenamiento (ROC-AUC).
  El conjunto de evaluación se mira una sola vez, para informar el resultado.
- El umbral de decisión también sale del entrenamiento (predicciones fuera de
  muestra), nunca del conjunto de evaluación.
- Solo el 4.7% de los créditos no se pagó a tiempo. Con esa proporción la
  exactitud no sirve (un modelo que siempre dice "pagó" acierta 95%). Las
  métricas se reportan para la clase que importa en riesgo de crédito: el
  impago (Pago_atiempo = 0).

Además de la comparación principal, el script hace dos pruebas de apoyo:
1. Con y sin `puntaje`, para documentar por qué se excluye (decisión del 2026-11-10).
2. Partición temporal en lugar de aleatoria, para ver si el resultado se sostiene
   cuando el modelo predice préstamos más nuevos que los de entrenamiento.

Uso:
    python src/model_training_evaluation.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # guarda las figuras en archivos, no abre ventanas

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ft_engineering import RANDOM_STATE, ROOT_DIR, build_datasets

# Los resultados van a una carpeta que Git ignora (ver .gitignore).
OUTPUT_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = OUTPUT_DIR / "figures"

N_SPLITS = 5
N_BOOTSTRAP = 500
BASELINE_NAME = "Baseline (siempre paga)"
PALETTE = {
    BASELINE_NAME: "#9e9e9e",
    "Regresión logística": "#1f77b4",
    "Random Forest": "#2ca02c",
    "HistGradientBoosting": "#ff7f0e",
    "XGBoost": "#d62728",
}


# ---------------------------------------------------------------------------
# Funciones que se repiten en todo el flujo
# ---------------------------------------------------------------------------
def summarize_classification(y_true, y_proba, threshold=0.5):
    """Calcula las métricas de un modelo de clasificación.

    y_proba es la probabilidad de pagar a tiempo (clase 1), que es lo que
    devuelve predict_proba. threshold es el umbral sobre la probabilidad de
    IMPAGO (1 - y_proba): si la supera, el crédito se marca como impago.

    Las métricas "_impago" miran a la clase 0, que es la minoritaria.
    """
    y_true = np.asarray(y_true)
    p_default = 1 - np.asarray(y_proba)
    flagged = p_default >= threshold  # True = el modelo anticipa un impago
    pred = np.where(flagged, 0, 1)  # misma codificación que Pago_atiempo

    is_default = y_true == 0
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc_impago": average_precision_score(is_default, p_default),
        "recall_impago": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "precision_impago": precision_score(y_true, pred, pos_label=0, zero_division=0),
        "f1_impago": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "threshold": threshold,
        # Matriz de confusión
        "impagos_detectados": int((is_default & flagged).sum()),
        "impagos_no_detectados": int((is_default & ~flagged).sum()),
        "falsas_alarmas": int((~is_default & flagged).sum()),
        "pagos_correctos": int((~is_default & ~flagged).sum()),
    }


def build_model(name, estimator, data, threshold=0.5):
    """Entrena un modelo con el conjunto de entrenamiento y lo evalúa en el de prueba.

    Devuelve un diccionario con el modelo ajustado, las probabilidades de
    prueba y las métricas (calculadas con summarize_classification).
    """
    model = clone(estimator).fit(data.X_train, data.y_train)
    proba_train = model.predict_proba(data.X_train)[:, 1]
    proba_test = model.predict_proba(data.X_test)[:, 1]

    metrics = summarize_classification(data.y_test, proba_test, threshold)
    metrics["roc_auc_train"] = roc_auc_score(data.y_train, proba_train)
    return {"name": name, "model": model, "proba_test": proba_test, "metrics": metrics}


def get_models(y_train):
    """Los modelos que se comparan, con ponderación para la clase minoritaria."""
    # XGBoost pondera la clase 1. Aquí la minoritaria es la 0, así que el peso
    # relativo de la clase 1 es menor que 1 (n_clase0 / n_clase1).
    xgb_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        BASELINE_NAME: DummyClassifier(strategy="prior"),
        "Regresión logística": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=xgb_weight,
            eval_metric="auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Validación y umbral
# ---------------------------------------------------------------------------
def cross_validate_model(estimator, data, n_splits=N_SPLITS):
    """Validación cruzada estratificada sobre el entrenamiento.

    El preprocesador se ajusta de nuevo en cada partición, con los datos de
    esa partición. Devuelve la media y la desviación de las métricas, y las
    probabilidades fuera de muestra (sirven para elegir el umbral).
    """
    X, y = data.X_train_raw, data.y_train
    pipe = Pipeline(
        [("prep", clone(data.preprocessor_template)), ("model", clone(estimator))]
    )
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof = pd.Series(np.nan, index=X.index)
    aucs, pr_aucs = [], []
    for train_idx, valid_idx in folds.split(X, y):
        fitted = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        oof.iloc[valid_idx] = proba
        y_valid = y.iloc[valid_idx]
        aucs.append(roc_auc_score(y_valid, proba))
        pr_aucs.append(average_precision_score(y_valid == 0, 1 - proba))

    return {
        "cv_auc_mean": np.mean(aucs),
        "cv_auc_std": np.std(aucs, ddof=1),
        "cv_pr_auc_mean": np.mean(pr_aucs),
        "oof_proba": oof,
    }


def choose_threshold(y_true, y_proba):
    """Umbral de probabilidad de impago que maximiza el F1 del impago."""
    y_true = np.asarray(y_true)
    p_default = 1 - np.asarray(y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true == 0, p_default)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[np.argmax(f1)])


def bootstrap_auc_interval(y_true, y_proba, n_boot=N_BOOTSTRAP):
    """Intervalo de 95% para el AUC de prueba, remuestreando las filas de prueba."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            values.append(roc_auc_score(y_true[idx], y_proba[idx]))
    return np.percentile(values, [2.5, 97.5])


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------
def evaluate_models(data, models, with_cv=True):
    """Entrena y evalúa todos los modelos. Devuelve (tabla resumen, resultados)."""
    results, rows = {}, []
    for name, estimator in models.items():
        threshold, cv = 0.5, {}
        if with_cv:
            cv = cross_validate_model(estimator, data)
            # El baseline no distingue clientes: con un umbral elegido marcaría a todos.
            # Se deja en 0.5 para que haga lo que dice su nombre (nunca anticipar impagos).
            if name != BASELINE_NAME:
                threshold = choose_threshold(data.y_train, cv["oof_proba"])

        result = build_model(name, estimator, data, threshold)
        low, high = bootstrap_auc_interval(data.y_test, result["proba_test"])
        results[name] = result

        rows.append(
            {
                "modelo": name,
                "auc_cv": cv.get("cv_auc_mean", np.nan),
                "auc_cv_std": cv.get("cv_auc_std", np.nan),
                "auc_train": result["metrics"]["roc_auc_train"],
                "auc_test": result["metrics"]["roc_auc"],
                "auc_test_ic_bajo": low,
                "auc_test_ic_alto": high,
                "pr_auc_impago": result["metrics"]["pr_auc_impago"],
                "recall_impago": result["metrics"]["recall_impago"],
                "precision_impago": result["metrics"]["precision_impago"],
                "f1_impago": result["metrics"]["f1_impago"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "accuracy": result["metrics"]["accuracy"],
                "umbral": threshold,
            }
        )
    return pd.DataFrame(rows).set_index("modelo"), results


def select_best_model(summary):
    """Mejor modelo según el AUC de validación cruzada (sin mirar la prueba)."""
    candidates = summary.drop(index=BASELINE_NAME, errors="ignore")
    best = candidates["auc_cv"].idxmax()
    margin = candidates.loc[best, "auc_cv_std"]
    ties = candidates.index[candidates["auc_cv"] >= candidates.loc[best, "auc_cv"] - margin]
    return best, [m for m in ties if m != best]


def to_markdown_table(df, float_format="{:.3f}"):
    """Convierte un DataFrame en una tabla Markdown (sin depender de tabulate)."""
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    divider = "|" + "---|" * (len(df.columns) + 1)
    lines = [header, divider]
    for idx, row in df.iterrows():
        cells = [float_format.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def build_report_table(summary):
    """Tabla resumen legible, ordenada por AUC de validación cruzada."""
    table = pd.DataFrame(
        {
            "AUC validación (media ± desv.)": [
                "—" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"
                for m, s in zip(summary["auc_cv"], summary["auc_cv_std"])
            ],
            "AUC prueba [IC 95%]": [
                f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
                for a, lo, hi in zip(
                    summary["auc_test"], summary["auc_test_ic_bajo"], summary["auc_test_ic_alto"]
                )
            ],
            "AUC entrenamiento": summary["auc_train"],
            "PR-AUC impago": summary["pr_auc_impago"],
            "Recall impago": summary["recall_impago"],
            "Precisión impago": summary["precision_impago"],
            "F1 impago": summary["f1_impago"],
            "Exactitud balanceada": summary["balanced_accuracy"],
            "Exactitud": summary["accuracy"],
            "Umbral impago": summary["umbral"],
        },
        index=summary.index,
    )
    table.index.name = "Modelo"
    return table.assign(_orden=summary["auc_cv"].fillna(-1)).sort_values(
        "_orden", ascending=False
    ).drop(columns="_orden")


# ---------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------
def _color(name):
    return PALETTE.get(name, "#555555")


def plot_roc_pr(results, y_test, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    is_default = np.asarray(y_test) == 0

    for name, res in results.items():
        if name == BASELINE_NAME:
            continue  # probabilidad constante: su curva no dice nada (la diagonal ya la representa)
        proba = res["proba_test"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, color=_color(name), lw=2, label=f"{name} (AUC {res['metrics']['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(is_default, 1 - proba)
        axes[1].plot(rec, prec, color=_color(name), lw=2, label=f"{name} (PR-AUC {res['metrics']['pr_auc_impago']:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=1, label="Azar (= baseline)")
    axes[0].set(
        title="Curva ROC (conjunto de prueba)",
        xlabel="Tasa de falsos positivos (positivo = pagó a tiempo)",
        ylabel="Tasa de verdaderos positivos",
    )
    axes[1].axhline(is_default.mean(), ls="--", color="black", lw=1, label="Azar (proporción de impagos)")
    axes[1].set(title="Precisión-recall del impago (conjunto de prueba)", xlabel="Recall", ylabel="Precisión")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(summary, path):
    models = [m for m in summary.index if m != BASELINE_NAME]
    sub = summary.loc[models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    width = 0.38
    axes[0].bar(x - width / 2, sub["auc_cv"], width, yerr=sub["auc_cv_std"], capsize=4, label="Validación cruzada", color="#4c78a8")
    axes[0].bar(
        x + width / 2,
        sub["auc_test"],
        width,
        yerr=[sub["auc_test"] - sub["auc_test_ic_bajo"], sub["auc_test_ic_alto"] - sub["auc_test"]],
        capsize=4,
        label="Prueba (IC 95%)",
        color="#f58518",
    )
    axes[0].axhline(0.5, ls="--", color="black", lw=1)
    axes[0].set(title="ROC-AUC", ylim=(0.4, 1.0), xticks=x, xticklabels=models)
    axes[0].legend()

    width = 0.26
    for i, (col, label) in enumerate(
        [("recall_impago", "Recall"), ("precision_impago", "Precisión"), ("f1_impago", "F1")]
    ):
        axes[1].bar(x + (i - 1) * width, sub[col], width, label=label)
    axes[1].set(title="Impago en el conjunto de prueba (umbral elegido en entrenamiento)", ylim=(0, 1), xticks=x, xticklabels=models)
    axes[1].legend()

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, path):
    names = [n for n in results if n != BASELINE_NAME]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.8))
    for ax, name in zip(np.atleast_1d(axes), names):
        m = results[name]["metrics"]
        matrix = np.array(
            [
                [m["impagos_detectados"], m["impagos_no_detectados"]],
                [m["falsas_alarmas"], m["pagos_correctos"]],
            ]
        )
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="black" if value < matrix.max() / 2 else "white")
        ax.set(
            title=f"{name}\numbral {m['threshold']:.2f}",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Anticipa impago", "Anticipa pago"],
            yticklabels=["Impagó", "Pagó"],
        )
        ax.tick_params(axis="x", rotation=15, labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_with_without_score(comparison, path):
    x = np.arange(len(comparison))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width / 2, comparison["auc_test_sin_puntaje"], width, label="Sin puntaje", color="#4c78a8")
    ax.bar(x + width / 2, comparison["auc_test_con_puntaje"], width, label="Con puntaje", color="#e45756")
    ax.axhline(0.5, ls="--", color="black", lw=1)
    ax.set(title="ROC-AUC de prueba con y sin `puntaje`", ylim=(0.4, 1.15), xticks=x, xticklabels=comparison.index)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    # 1. Comparación principal (sin puntaje, partición aleatoria estratificada)
    data = build_datasets(include_score=False)
    print(f"Entrenamiento: {data.X_train.shape} | Prueba: {data.X_test.shape}")
    print(f"Impagos en prueba: {int((data.y_test == 0).sum())} de {len(data.y_test)}\n")

    models = get_models(data.y_train)
    summary, results = evaluate_models(data, models)
    report = build_report_table(summary)

    print("=== Tabla resumen (sin `puntaje`) ===")
    print(report.to_string(float_format=lambda v: f"{v:.3f}"))
    report.to_csv(OUTPUT_DIR / "model_summary.csv", float_format="%.4f")
    (OUTPUT_DIR / "model_summary.md").write_text(to_markdown_table(report), encoding="utf-8")

    best_name, ties = select_best_model(summary)
    print(f"\nMejor modelo por AUC de validación cruzada: {best_name}")
    if ties:
        print(f"Dentro de una desviación estándar del mejor: {', '.join(ties)}")

    plot_roc_pr(results, data.y_test, FIGURES_DIR / "roc_pr_curves.png")
    plot_metric_comparison(summary, FIGURES_DIR / "metric_comparison.png")
    plot_confusion_matrices(results, FIGURES_DIR / "confusion_matrices.png")

    # Se guarda el pipeline completo (preprocesador + modelo), ajustado con el entrenamiento.
    final_pipeline = Pipeline(
        [("preprocessor", data.preprocessor), ("model", results[best_name]["model"])]
    )
    joblib.dump(final_pipeline, OUTPUT_DIR / "best_model.joblib")

    # 2. Con y sin `puntaje`
    data_score = build_datasets(include_score=True)
    summary_score, _ = evaluate_models(data_score, get_models(data_score.y_train))
    comparison = pd.DataFrame(
        {
            "auc_test_sin_puntaje": summary["auc_test"],
            "auc_test_con_puntaje": summary_score["auc_test"],
            "auc_cv_sin_puntaje": summary["auc_cv"],
            "auc_cv_con_puntaje": summary_score["auc_cv"],
        }
    ).drop(index=BASELINE_NAME)
    comparison["diferencia_auc_test"] = comparison["auc_test_con_puntaje"] - comparison["auc_test_sin_puntaje"]
    print("\n=== Con y sin `puntaje` ===")
    print(comparison.to_string(float_format=lambda v: f"{v:.3f}"))
    comparison.to_csv(OUTPUT_DIR / "score_comparison.csv", float_format="%.4f")
    plot_with_without_score(comparison, FIGURES_DIR / "with_without_score.png")

    # 3. Partición temporal (sin validación cruzada: solo AUC y PR-AUC de prueba)
    data_time = build_datasets(split_mode="temporal")
    summary_time, _ = evaluate_models(data_time, get_models(data_time.y_train), with_cv=False)
    temporal = pd.DataFrame(
        {
            "auc_test_aleatoria": summary["auc_test"],
            "auc_test_temporal": summary_time["auc_test"],
            "pr_auc_aleatoria": summary["pr_auc_impago"],
            "pr_auc_temporal": summary_time["pr_auc_impago"],
        }
    ).drop(index=BASELINE_NAME)
    print(f"\n=== Partición temporal (impagos en prueba: {int((data_time.y_test == 0).sum())}) ===")
    print(temporal.to_string(float_format=lambda v: f"{v:.3f}"))
    temporal.to_csv(OUTPUT_DIR / "temporal_split_comparison.csv", float_format="%.4f")

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
