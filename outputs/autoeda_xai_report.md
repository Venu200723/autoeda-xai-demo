# AutoEDA + XAI Demo Report

**Generated:** 2026-09-18 09:03  
**Rows × cols:** 403 × 12  
**Target:** `churn`

## Auto EDA figures

- `/workspace/career-portfolio/autoeda-xai-demo/outputs/autoeda_distributions.png`
- `/workspace/career-portfolio/autoeda-xai-demo/outputs/autoeda_target_balance.png`
- `/workspace/career-portfolio/autoeda-xai-demo/outputs/autoeda_corr.png`

## Model

- Train/test: **302** / **101**
- Accuracy: **0.7426**
- F1 (weighted): **0.7003**

## Explainability (`permutation`)

- Figure: `/workspace/career-portfolio/autoeda-xai-demo/outputs/xai_permutation_importance.png`

| Feature | Importance |
|---|---:|
| `monthly_charges` | 0.0040 |
| `contract` | 0.0040 |
| `senior_citizen` | 0.0040 |
| `tenure_months` | -0.0079 |
| `internet_service` | -0.0079 |
| `total_charges` | -0.0099 |
| `tech_support` | -0.0099 |
| `dependents` | -0.0099 |
| `payment_method` | -0.0257 |
| `partner` | -0.0396 |

---
_autoeda-xai-demo — load CSV → Auto EDA → model → SHAP/permutation → report._