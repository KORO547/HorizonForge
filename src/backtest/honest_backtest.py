"""Честный walk-forward бэктест дневной модели для акций.

Принципы (чем он отличается от старых бэктестов):
  1. Признаки считаются только по данным, известным на закрытии дня t.
  2. Цель и доходность — следующий день: fwd_ret = Close[t+1] / Close[t] - 1.
  3. Модель каждый месяц переобучается ТОЛЬКО на прошлом (скользящее окно 3 года);
     прогнозы делаются на следующий месяц, которого модель не видела.
  4. Каждый день покупаем top-K акций по вероятности роста, держим 1 день,
     платим комиссию за оборот.
  5. Сравнение с SPY (buy & hold) и равновесным портфелем всей вселенной.
  6. Для наглядности считается и «режим с утечкой» как в старом бэктесте:
     те же выборы, но с доходностью ТЕКУЩЕГО дня (ret_1) — так получались +112%.

Ограничения: вселенная — сегодняшний список акций (survivorship bias, слегка
завышает результат); исполнение по цене закрытия; без проскальзывания.

Запуск из корня проекта (после download_long_history.py):
    python src/backtest/honest_backtest.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data/raw/stocks/long"
OUT_DIR = "results/honest_backtest"
BENCHMARK = "SPY"

TRAIN_YEARS = 3          # длина скользящего окна обучения
MIN_TRAIN_DAYS = 500     # минимум торговых дней истории перед первым тестом
TOP_K = [5, 20]          # размеры портфеля
COMMISSION = 0.001       # 0.1% за каждую сторону сделки

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "rsi_14", "macd_hist_n", "bb_pos", "atr_n",
    "dist_ema21", "dist_ema50", "dist_ema200",
    "vol_10", "vol_60", "volume_ratio", "price_range", "dayofweek",
    "mkt_ret_1", "mkt_ret_5", "rel_ret_5", "rel_ret_20",
    "rank_ret_5", "rank_ret_20",
]


# ---------------------------------------------------------------- features
def ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = c.pct_change()
    for n in (5, 10, 20):
        out[f"ret_{n}"] = c.pct_change(n)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (12, 21, 26, 50, 200)}
    macd = ema[12] - ema[26]
    out["macd_hist_n"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pos"] = (c - sma20) / (2 * std20)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_n"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c

    for n in (21, 50, 200):
        out[f"dist_ema{n}"] = c / ema[n] - 1
    # EMA-200 нужно минимум ~200 дней истории, иначе признак бессмысленный
    out.loc[out.index[:200], "dist_ema200"] = np.nan

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_60"] = out["ret_1"].rolling(60).std()
    out["volume_ratio"] = v / v.rolling(20).mean()
    out["price_range"] = (h - l) / c
    out["dayofweek"] = df["timestamp"].dt.dayofweek

    out["fwd_ret"] = c.shift(-1) / c - 1      # доходность СЛЕДУЮЩЕГО дня (цель)
    out["timestamp"] = df["timestamp"]
    return out


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    bench = None
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith("_1d.csv"):
            continue
        t = f.replace("_1d.csv", "")
        raw = pd.read_csv(f"{DATA_DIR}/{f}", parse_dates=["timestamp"])
        raw = raw.dropna(subset=["Close"]).sort_values("timestamp").reset_index(drop=True)
        if t == BENCHMARK:
            bench = raw.set_index("timestamp")["Close"].pct_change().shift(-1)  # fwd ret SPY
            continue
        feats = ticker_features(raw)
        feats["ticker"] = t
        frames.append(feats)

    panel = pd.concat(frames, ignore_index=True)

    # Кросс-секционные признаки (по всем акциям в один и тот же день — известны на закрытии t)
    g = panel.groupby("timestamp")
    panel["mkt_ret_1"] = g["ret_1"].transform("mean")
    panel["mkt_ret_5"] = g["ret_5"].transform("mean")
    panel["rel_ret_5"] = panel["ret_5"] - panel["mkt_ret_5"]
    panel["rel_ret_20"] = panel["ret_20"] - g["ret_20"].transform("mean")
    panel["rank_ret_5"] = g["ret_5"].rank(pct=True)
    panel["rank_ret_20"] = g["ret_20"].rank(pct=True)

    panel = panel.dropna(subset=FEATURES).copy()
    panel[FEATURES] = panel[FEATURES].astype("float32")
    panel = panel.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    return panel, bench


# ---------------------------------------------------------------- walk-forward
def walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    dates = np.sort(panel["timestamp"].unique())
    first_test = pd.Timestamp(dates[MIN_TRAIN_DAYS])
    months = pd.period_range(first_test, pd.Timestamp(dates[-1]), freq="M")

    preds = []
    for i, m in enumerate(months, 1):
        m_start, m_end = m.start_time, m.end_time
        train = panel[(panel["timestamp"] < m_start)
                      & (panel["timestamp"] >= m_start - pd.DateOffset(years=TRAIN_YEARS))
                      & panel["fwd_ret"].notna()]
        test = panel[(panel["timestamp"] >= m_start) & (panel["timestamp"] <= m_end)]
        if test.empty:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            min_child_samples=500, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, verbose=-1,
        )
        model.fit(train[FEATURES], (train["fwd_ret"] > 0).astype(int))

        p = test[["timestamp", "ticker", "fwd_ret", "ret_1"]].copy()
        p["prob"] = model.predict_proba(test[FEATURES])[:, 1]
        preds.append(p)
        if i % 12 == 0 or i == len(months):
            print(f"  обучено {i}/{len(months)} месяцев (до {m})")

    return pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------- portfolio
def portfolio_returns(preds: pd.DataFrame, k: int) -> pd.DataFrame:
    """Ежедневно top-k по prob, равные веса, держим 1 день."""
    rows = []
    prev = set()
    for ts, day in preds.groupby("timestamp", sort=True):
        picks = day.nlargest(k, "prob")
        cur = set(picks["ticker"])
        # оборот: доля портфеля, которую пришлось купить + продать
        turnover = 1.0 if not prev else (len(cur - prev) + len(prev - cur)) / k
        rows.append({
            "timestamp": ts,
            "gross": picks["fwd_ret"].mean(),      # честно: доходность следующего дня
            "leaky": picks["ret_1"].mean(),        # как в старом бэктесте: уже случившийся день
            "turnover": turnover,
        })
        prev = cur
    r = pd.DataFrame(rows).set_index("timestamp")
    r["net"] = r["gross"] - r["turnover"] * COMMISSION
    return r


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    dd = (eq / eq.cummax() - 1).min()
    return {
        "Доходность, %": (eq.iloc[-1] - 1) * 100,
        "CAGR, %": (eq.iloc[-1] ** (1 / years) - 1) * 100,
        "Волатильность, %": r.std() * np.sqrt(252) * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "Макс. просадка, %": dd * 100,
        "Дней в плюс, %": (r > 0).mean() * 100,
    }


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Честный walk-forward бэктест ===\n")

    panel, bench = load_panel()
    print(f"Панель: {len(panel):,} строк, {panel['ticker'].nunique()} акций, "
          f"{panel['timestamp'].min().date()} … {panel['timestamp'].max().date()}\n")

    print("Walk-forward обучение (ежемесячно, только на прошлом):")
    preds = walk_forward(panel)
    ev = preds.dropna(subset=["fwd_ret"])
    print(f"\nТестовый период (модель его не видела): "
          f"{ev['timestamp'].min().date()} … {ev['timestamp'].max().date()} "
          f"({ev['timestamp'].nunique()} дней)\n")

    # --- качество прогноза
    y = (ev["fwd_ret"] > 0).astype(int)
    auc = roc_auc_score(y, ev["prob"])
    by_year = ev.assign(y=y).groupby(ev["timestamp"].dt.year).apply(
        lambda d: roc_auc_score(d["y"], d["prob"]) if d["y"].nunique() > 1 else np.nan)
    ic = ev.groupby("timestamp").apply(
        lambda d: d["prob"].rank().corr(d["fwd_ret"].rank()))
    ic_t = ic.mean() / ic.std() * np.sqrt(len(ic))

    print("Качество прогноза вне выборки:")
    print(f"  AUC (направление след. дня): {auc:.4f}   (0.5 = монетка; старый дашборд: 0.738)")
    print("  AUC по годам: " + ", ".join(f"{y_}: {v:.3f}" for y_, v in by_year.items()))
    print(f"  Rank IC (ранжирование акций): {ic.mean():+.4f}, t-stat {ic_t:+.2f}  "
          f"(|t| > 2 — сигнал статистически заметен)\n")

    # --- портфели
    ew = ev.groupby("timestamp")["fwd_ret"].mean()
    table = {}
    daily = pd.DataFrame(index=ew.index)
    daily["Равновесный (все акции)"] = ew
    daily["SPY buy & hold"] = bench.reindex(ew.index)
    for k in TOP_K:
        pr = portfolio_returns(ev, k).reindex(ew.index)
        daily[f"Модель top-{k} (с комиссией)"] = pr["net"]
        daily[f"Модель top-{k} (без комиссии)"] = pr["gross"]
        if k == TOP_K[0]:
            daily[f"Утечка top-{k} (как старый бэктест)"] = pr["leaky"]
        print(f"top-{k}: средний дневной оборот {pr['turnover'].mean():.0%}, "
              f"комиссии съедают {pr['turnover'].mean() * COMMISSION * 252 * 100:.1f}% в год")

    for col in daily.columns:
        table[col] = metrics(daily[col])
    summary = pd.DataFrame(table).T.round(2)

    pd.set_option("display.width", 200)
    print("\nРезультаты за тестовый период:")
    print(summary.to_string())

    summary.to_csv(f"{OUT_DIR}/summary.csv", encoding="utf-8-sig")
    daily.to_csv(f"{OUT_DIR}/daily_returns.csv")
    by_year.rename("auc").to_csv(f"{OUT_DIR}/auc_by_year.csv")

    # --- график (без «утечки» — она уходит в небеса и ломает масштаб)
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in daily.columns:
        if col.startswith("Утечка") or "без комиссии" in col:
            continue
        ax.plot((1 + daily[col].fillna(0)).cumprod(), label=col)
    ax.set_yscale("log")
    ax.set_title("Честный walk-forward: рост $1 (лог. шкала)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/equity.png", dpi=120)
    print(f"\nСохранено в {OUT_DIR}/ (summary.csv, daily_returns.csv, auc_by_year.csv, equity.png)")


if __name__ == "__main__":
    main()
