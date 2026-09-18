# AutoEDA + XAI Demo

Compact demo script: **load CSV → Auto EDA plots → train a simple model → SHAP (or permutation importance fallback) → Markdown/HTML report**.

## How to run

```bash
cd autoeda-xai-demo
pip install -r requirements.txt
# shap is optional; without it the script falls back to sklearn permutation importance
python run_autoeda_xai.py --csv data/sample.csv --target churn --out outputs
```

## Outputs

- `outputs/autoeda_distributions.png`
- `outputs/autoeda_target_balance.png`
- `outputs/autoeda_corr.png`
- `outputs/xai_shap_importance.png` *or* `xai_permutation_importance.png`
- `outputs/autoeda_xai_report.md`
- `outputs/autoeda_xai_report.html`

## Resume one-liner

Shipped an **AutoEDA + explainability demo** that produces distribution/correlation plots, a baseline classifier, and SHAP/permutation importance in a single HTML/Markdown report.

## Suggested GitHub repo name

`autoeda-xai-demo`
