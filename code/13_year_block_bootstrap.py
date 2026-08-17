"""Year-block bootstrap intervals for pre-specified B5 versus B1 OOT gains."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "work" / "prediction" / "results"
rng = np.random.default_rng(20260806)
x = pd.read_csv(RES / "rolling_oot_metrics_by_year.csv")
specs = [
    ("forward_cites_5y", "poisson_deviance", "lower_is_better"),
    ("semantic_reuse_5y_log1p", "r2", "higher_is_better"),
    ("top5_impact", "ndcg_at_k", "higher_is_better"),
    ("outside_field_cites_5y_strict", "poisson_deviance", "lower_is_better"),
]
rows=[]
for outcome, metric, direction in specs:
    d=x.loc[(x.outcome.eq(outcome)) & (x.model.isin(["B1","B5"])),["test_year","model",metric]]
    w=d.pivot(index="test_year",columns="model",values=metric).dropna()
    delta=(w.B1-w.B5) if direction=="lower_is_better" else (w.B5-w.B1)
    draws=np.array([rng.choice(delta.to_numpy(), size=len(delta), replace=True).mean() for _ in range(500)])
    rows.append({"outcome":outcome,"metric":metric,"comparison":"B5 improvement over B1","n_test_years":len(delta),"mean_improvement":delta.mean(),"ci95_lower":np.quantile(draws,.025),"ci95_upper":np.quantile(draws,.975)})
pd.DataFrame(rows).to_csv(RES / "b5_vs_b1_year_block_bootstrap.csv",index=False,encoding="utf-8-sig")
