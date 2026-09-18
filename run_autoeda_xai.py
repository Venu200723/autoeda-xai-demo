#!/usr/bin/env python3
"""Auto EDA → simple model → SHAP/permutation importance → markdown/HTML report."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # light clean
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[col] = coerced
    df = df.dropna(how="all")
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "unknown")
    return df


def auto_eda(df: pd.DataFrame, target: str, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    figs: list[str] = []
    numeric = df.select_dtypes(include=[np.number])
    cols = [c for c in numeric.columns if c != target][:4]
    if cols:
        fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 3))
        if len(cols) == 1:
            axes = [axes]
        for ax, c in zip(axes, cols):
            ax.hist(df[c], bins=20, color="#4C78A8", edgecolor="white")
            ax.set_title(c)
        fig.suptitle("Auto EDA — numeric distributions")
        fig.tight_layout()
        p = out / "autoeda_distributions.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        figs.append(str(p))

    if target in df.columns and df[target].nunique() <= 10:
        fig, ax = plt.subplots(figsize=(4, 3))
        df[target].value_counts().plot(kind="bar", ax=ax, color="#F58518")
        ax.set_title(f"Target balance: {target}")
        fig.tight_layout()
        p = out / "autoeda_target_balance.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        figs.append(str(p))

    if numeric.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cax = ax.imshow(numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric.columns)))
        ax.set_yticks(range(len(numeric.columns)))
        ax.set_xticklabels(numeric.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(numeric.columns, fontsize=7)
        fig.colorbar(cax, fraction=0.046)
        fig.tight_layout()
        p = out / "autoeda_corr.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        figs.append(str(p))
    return figs


def train_and_explain(df: pd.DataFrame, target: str, out: Path) -> dict:
    y = df[target]
    X = df.drop(columns=[target])
    drop_ids = [c for c in X.columns if c.lower() in {"id", "customer_id", "index"}]
    X = X.drop(columns=drop_ids, errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
    }

    # Prefer SHAP if installed; else permutation importance
    method = "permutation"
    importances: list[tuple[str, float]] = []
    try:
        import shap  # type: ignore

        # Explain on transformed matrix for tree model
        X_test_t = pipe.named_steps["pre"].transform(X_test)
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        sv = explainer.shap_values(X_test_t)
        if isinstance(sv, list):
            # classification: take positive class
            vals = np.abs(sv[1]).mean(axis=0) if len(sv) > 1 else np.abs(sv[0]).mean(axis=0)
        else:
            vals = np.abs(sv).mean(axis=0)
        # feature names
        try:
            names = pipe.named_steps["pre"].get_feature_names_out()
        except Exception:
            names = [f"f{i}" for i in range(len(vals))]
        importances = sorted(zip(names, vals), key=lambda x: x[1], reverse=True)[:12]
        method = "shap"
        # summary-style bar
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [str(n) for n, _ in reversed(importances)]
        means = [float(v) for _, v in reversed(importances)]
        ax.barh(labels, means, color="#B279A2")
        ax.set_title("SHAP mean |value|")
        fig.tight_layout()
        fig_path = out / "xai_shap_importance.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
    except Exception:
        result = permutation_importance(
            pipe, X_test, y_test, n_repeats=5, random_state=42, scoring="accuracy"
        )
        importances = sorted(
            zip(X_test.columns, result.importances_mean),
            key=lambda x: x[1],
            reverse=True,
        )[:12]
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [str(n) for n, _ in reversed(importances)]
        means = [float(v) for _, v in reversed(importances)]
        ax.barh(labels, means, color="#54A24B")
        ax.set_title("Permutation importance")
        fig.tight_layout()
        fig_path = out / "xai_permutation_importance.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

    return {
        "metrics": metrics,
        "method": method,
        "importances": [(str(n), float(v)) for n, v in importances],
        "figure": str(fig_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def write_report(
    df: pd.DataFrame,
    target: str,
    eda_figs: list[str],
    xai: dict,
    out: Path,
) -> tuple[Path, Path]:
    md = out / "autoeda_xai_report.md"
    lines = [
        "# AutoEDA + XAI Demo Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Rows × cols:** {df.shape[0]} × {df.shape[1]}  ",
        f"**Target:** `{target}`",
        "",
        "## Auto EDA figures",
        "",
    ]
    for f in eda_figs:
        lines.append(f"- `{f}`")
    lines += [
        "",
        "## Model",
        "",
        f"- Train/test: **{xai['n_train']}** / **{xai['n_test']}**",
        f"- Accuracy: **{xai['metrics']['accuracy']:.4f}**",
        f"- F1 (weighted): **{xai['metrics']['f1_weighted']:.4f}**",
        "",
        f"## Explainability (`{xai['method']}`)",
        "",
        f"- Figure: `{xai['figure']}`",
        "",
        "| Feature | Importance |",
        "|---|---:|",
    ]
    for name, val in xai["importances"]:
        lines.append(f"| `{name}` | {val:.4f} |")
    lines += [
        "",
        "---",
        "_autoeda-xai-demo — load CSV → Auto EDA → model → SHAP/permutation → report._",
    ]
    text = "\n".join(lines)
    md.write_text(text, encoding="utf-8")

    html = out / "autoeda_xai_report.html"
    body = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>\n")
    html.write_text(
        "<html><head><meta charset='utf-8'><title>AutoEDA XAI Report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;"
        "line-height:1.55}</style></head><body>"
        f"{body}</body></html>",
        encoding="utf-8",
    )
    return md, html


def main() -> int:
    p = argparse.ArgumentParser(description="AutoEDA + XAI demo")
    p.add_argument("--csv", default="data/sample.csv")
    p.add_argument("--target", default="churn")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    csv_path = (root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not in {list(df.columns)}")

    eda_figs = auto_eda(df, args.target, out)
    xai = train_and_explain(df, args.target, out)
    md, html = write_report(df, args.target, eda_figs, xai, out)

    print("AutoEDA+XAI complete")
    print(f"  metrics : {xai['metrics']}")
    print(f"  method  : {xai['method']}")
    print(f"  report  : {md}")
    print(f"  html    : {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
