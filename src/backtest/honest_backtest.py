"""Честный walk-forward бэктест стратегий выбора акций.

Принципы:
  1. Признаки считаются только по данным, известным на закрытии дня t.
  2. Модель каждый месяц переобучается ТОЛЬКО на прошлом (скользящее окно 3 года).
     В обучение попадают лишь примеры, чей итоговый период уже закончился до
     начала тестового месяца (purge) — модель не подглядывает в будущее.
  3. Цель: обгонит ли акция медианную акцию за следующие H дней.
  4. Доходность портфеля считается по дням: позиции, выбранные на закрытии t,
     зарабатывают Close[t+1]/Close[t]-1. Комиссия 0.1% за каждую сторону сделки.
  5. Портфель пересобирается раз в R дней; «буфер» B — уже купленная акция
     продаётся, только если выпала из топ-B (снижает оборот).
  6. Сравнение с SPY (buy & hold) и равновесным портфелем всех акций.

Ограничения: вселенная — сегодняшний список акций (survivorship bias, слегка
завышает результат); исполнение по цене закрытия; без проскальзывания;
веса внутри портфеля условно равные каждый день.

Запуск из корня проекта (после download_long_history.py):
    python src/backtest/honest_backtest.py
Результаты: results/honest_backtest/ (summary.csv, yearly.csv, equity.png, dash/*.json)
"""
import json
import os
import sys
from datetime import datetime

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
MIN_TRAIN_DAYS = 500     # торговых дней истории перед первым тестом
COMMISSION = 0.001       # 0.1% за каждую сторону сделки

CONFIGS = [
    {"id": "A", "name": "Ежедневно, топ-20", "horizon": 1, "rebalance": 1, "k": 20, "buffer": 20},
    {"id": "B", "name": "Раз в неделю, топ-20", "horizon": 5, "rebalance": 5, "k": 20, "buffer": 20},
    {"id": "C", "name": "Раз в неделю, топ-20, буфер 40", "horizon": 5, "rebalance": 5, "k": 20, "buffer": 40},
    {"id": "D", "name": "Раз в месяц, топ-20, буфер 40", "horizon": 20, "rebalance": 20, "k": 20, "buffer": 40},
]
HORIZONS = sorted({c["horizon"] for c in CONFIGS})

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
    out.loc[out.index[:200], "dist_ema200"] = np.nan   # EMA-200 без истории бессмысленна

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_60"] = out["ret_1"].rolling(60).std()
    out["volume_ratio"] = v / v.rolling(20).mean()
    out["price_range"] = (h - l) / c
    out["dayofweek"] = df["timestamp"].dt.dayofweek

    # Будущее — только для целей и оценки, НЕ признаки
    for hz in sorted(set(HORIZONS) | {1}):
        out[f"fwd_{hz}"] = c.shift(-hz) / c - 1
        out[f"label_end_{hz}"] = df["timestamp"].shift(-hz)   # когда итог периода станет известен
    out["timestamp"] = df["timestamp"]
    return out


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    frames, bench = [], None
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith("_1d.csv"):
            continue
        t = f.replace("_1d.csv", "")
        raw = pd.read_csv(f"{DATA_DIR}/{f}", parse_dates=["timestamp"])
        raw = raw.dropna(subset=["Close"]).sort_values("timestamp").reset_index(drop=True)
        if t == BENCHMARK:
            bench = raw.set_index("timestamp")["Close"].pct_change().shift(-1)
            continue
        feats = ticker_features(raw)
        feats["ticker"] = t
        frames.append(feats)

    panel = pd.concat(frames, ignore_index=True)
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
def walk_forward(panel: pd.DataFrame, hz: int) -> pd.DataFrame:
    """Ежемесячно обучает модель на прошлом и предсказывает следующий месяц."""
    fwd, end = f"fwd_{hz}", f"label_end_{hz}"
    # цель: обгонит ли акция медиану за hz дней
    panel["_y"] = (panel[fwd] > panel.groupby("timestamp")[fwd].transform("median")).astype(int)

    dates = np.sort(panel["timestamp"].unique())
    months = pd.period_range(pd.Timestamp(dates[MIN_TRAIN_DAYS]), pd.Timestamp(dates[-1]), freq="M")

    preds = []
    for i, m in enumerate(months, 1):
        m_start, m_end = m.start_time, m.end_time
        train = panel[(panel["timestamp"] >= m_start - pd.DateOffset(years=TRAIN_YEARS))
                      & (panel[end] < m_start)          # purge: итог известен до теста
                      & panel[fwd].notna()]
        test = panel[(panel["timestamp"] >= m_start) & (panel["timestamp"] <= m_end)]
        if test.empty:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            min_child_samples=500, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, verbose=-1,
        )
        model.fit(train[FEATURES], train["_y"])
        p = test[list(dict.fromkeys(["timestamp", "ticker", "fwd_1", fwd]))].copy()
        p["prob"] = model.predict_proba(test[FEATURES])[:, 1]
        preds.append(p)
        if i % 24 == 0 or i == len(months):
            print(f"    {i}/{len(months)} месяцев")
    panel.drop(columns="_y", inplace=True)
    return pd.concat(preds, ignore_index=True)


def prediction_quality(preds: pd.DataFrame, hz: int) -> dict:
    fwd = f"fwd_{hz}"
    ev = preds.dropna(subset=[fwd])
    y = (ev[fwd] > ev.groupby("timestamp")[fwd].transform("median")).astype(int)
    auc = roc_auc_score(y, ev["prob"])
    # Rank IC по непересекающимся датам (каждая hz-я), иначе t-stat завышен
    dates = np.sort(ev["timestamp"].unique())[::hz]
    sub = ev[ev["timestamp"].isin(dates)]
    ic = sub.groupby("timestamp")[["prob", fwd]].apply(
        lambda d: d["prob"].rank().corr(d[fwd].rank()))
    return {"horizon": hz, "auc": round(auc, 4), "ic": round(ic.mean(), 4),
            "ic_t": round(ic.mean() / ic.std() * np.sqrt(len(ic)), 2)}


# ---------------------------------------------------------------- portfolio
def simulate(preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    k, buf, reb = cfg["k"], cfg["buffer"], cfg["rebalance"]
    by_day = {ts: d.sort_values("prob", ascending=False) for ts, d in preds.groupby("timestamp")}
    holdings: set[str] = set()
    rows = []
    for i, ts in enumerate(sorted(by_day)):
        day = by_day[ts]
        turnover = 0.0
        if i % reb == 0:
            ranked = list(day["ticker"])
            keep = holdings & set(ranked[:buf])
            new = set(keep)
            for t in ranked:
                if len(new) >= k:
                    break
                new.add(t)
            turnover = (len(holdings - new) + len(new - holdings)) / k
            holdings = new
        held = day[day["ticker"].isin(holdings)]
        rows.append({"timestamp": ts, "gross": held["fwd_1"].mean(), "turnover": turnover})
    r = pd.DataFrame(rows).set_index("timestamp")
    r["net"] = r["gross"] - r["turnover"] * COMMISSION
    return r


def metrics(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    m = {
        "Доходность, %": (eq.iloc[-1] - 1) * 100,
        "CAGR, %": (eq.iloc[-1] ** (1 / years) - 1) * 100,
        "Волатильность, %": r.std() * np.sqrt(252) * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "Макс. просадка, %": (eq / eq.cummax() - 1).min() * 100,
    }
    if turnover is not None:
        yearly_turn = turnover.reindex(r.index).sum() / years
        m["Оборот в год, x"] = yearly_turn / 2          # раз в год портфель меняется целиком
        m["Комиссии в год, %"] = yearly_turn * COMMISSION * 100
    else:
        m["Оборот в год, x"] = 0.0
        m["Комиссии в год, %"] = 0.0
    return m


# ---------------------------------------------------------------- dashboard export
def export_dashboard_files(summary, yearly, eq, dd, sample, ids, quality, period, universe):
    """Пишет results/honest_backtest/dash/*.json для визуального отчёта."""
    out = f"{OUT_DIR}/dash"
    os.makedirs(out, exist_ok=True)
    with open(__file__, encoding="utf-8") as f:
        code = f.read()
    meta = [
        {"label": "honest_backtest.py", "value": code, "type": "code", "lang": "python"},
        {"label": "Input", "value": f"{DATA_DIR}: дневные цены Yahoo Finance (10 лет), {universe} акций + SPY"},
        {"label": "Test period", "value": f"{period[0]} … {period[1]}"},
        {"label": "Generated", "value": datetime.now().isoformat(timespec="minutes")},
    ]
    num = lambda v, d=2: None if pd.isna(v) else round(float(v), d)

    def write(name, rows):
        with open(f"{out}/{name}.json", "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "meta": meta}, f, ensure_ascii=False)

    cfg_by_name = {c["name"]: c for c in CONFIGS}
    rows = []
    for name, sid in ids.items():
        s = summary.loc[name]
        gross = summary.loc[name + " (без комиссии)"] if name in cfg_by_name else s
        c = cfg_by_name.get(name)
        rows.append({
            "id": sid, "strategy": name, "kind": "model" if c else "benchmark",
            "rebalance_days": c["rebalance"] if c else None,
            "horizon_days": c["horizon"] if c else None,
            "cagr_net": num(s["CAGR, %"]), "cagr_gross": num(gross["CAGR, %"]),
            "total_net": num(s["Доходность, %"], 1), "sharpe": num(s["Sharpe"]),
            "max_dd": num(s["Макс. просадка, %"], 1), "vol": num(s["Волатильность, %"], 1),
            "turnover_x": num(s["Оборот в год, x"], 1), "costs_pct": num(s["Комиссии в год, %"]),
        })
    write("summary", rows)

    write("equity", [
        {"date": str(d.date()), "series_id": ids[c], "value": num(eq.loc[d, c], 4),
         "drawdown": num(dd.loc[d, c] * 100, 2)}
        for c in ids for d in sample
    ])
    write("yearly", [
        {"year": int(y), "series_id": ids[c], "ret": num(yearly.loc[y, c], 1)}
        for c in ids for y in yearly.index
    ])
    write("quality", [
        {"horizon_days": q["horizon"], "auc": q["auc"], "rank_ic": q["ic"], "t_stat": q["ic_t"]}
        for q in quality
    ])


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Честный walk-forward бэктест: варианты удержания ===\n")
    panel, bench = load_panel()
    print(f"Панель: {len(panel):,} строк, {panel['ticker'].nunique()} акций, "
          f"{panel['timestamp'].min().date()} … {panel['timestamp'].max().date()}\n")

    preds_by_h, quality = {}, []
    for hz in HORIZONS:
        print(f"  Обучение моделей на горизонт {hz} дн.:")
        preds_by_h[hz] = walk_forward(panel, hz)
        quality.append(prediction_quality(preds_by_h[hz], hz))
    print()

    base = preds_by_h[HORIZONS[0]].dropna(subset=["fwd_1"])
    index = pd.DatetimeIndex(sorted(base["timestamp"].unique()))
    daily = pd.DataFrame(index=index)
    turns = {}
    for cfg in CONFIGS:
        sim = simulate(preds_by_h[cfg["horizon"]].dropna(subset=["fwd_1"]), cfg).reindex(index)
        daily[cfg["name"]] = sim["net"]
        daily[cfg["name"] + " (без комиссии)"] = sim["gross"]
        turns[cfg["name"]] = sim["turnover"]
    daily["SPY buy & hold"] = bench.reindex(index)
    daily["Равновесный (все акции)"] = base.groupby("timestamp")["fwd_1"].mean().reindex(index)

    summary = pd.DataFrame({c: metrics(daily[c], turns.get(c)) for c in daily.columns}).T.round(2)
    yearly = daily.groupby(daily.index.year).apply(lambda d: (1 + d).prod() - 1).mul(100).round(1)

    pd.set_option("display.width", 220)
    print("Качество прогноза вне выборки (цель — обогнать медиану):")
    for q in quality:
        print(f"  горизонт {q['horizon']:>2} дн.: AUC {q['auc']:.4f}, Rank IC {q['ic']:+.4f} (t {q['ic_t']:+.2f})")
    print(f"\nТестовый период: {index.min().date()} … {index.max().date()} ({len(index)} дней)\n")
    print(summary.to_string())
    print("\nДоходность по годам, %:")
    print(yearly.T.to_string())

    summary.to_csv(f"{OUT_DIR}/summary.csv", encoding="utf-8-sig")
    yearly.to_csv(f"{OUT_DIR}/yearly.csv", encoding="utf-8-sig")
    daily.to_csv(f"{OUT_DIR}/daily_returns.csv")

    # --- данные для визуального отчёта (JSON {rows, meta}; meta несёт код этого скрипта)
    main_cols = [c["name"] for c in CONFIGS] + ["SPY buy & hold", "Равновесный (все акции)"]
    ids = {c["name"]: c["id"] for c in CONFIGS} | {"SPY buy & hold": "SPY", "Равновесный (все акции)": "EW"}
    eq = (1 + daily[main_cols].fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    sample = eq.index[::5].append(eq.index[-1:]).unique()
    export_dashboard_files(
        summary=summary, yearly=yearly, eq=eq, dd=dd, sample=sample, ids=ids,
        quality=quality, period=(index.min().date(), index.max().date()),
        universe=int(panel["ticker"].nunique()),
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in main_cols:
        ax.plot(eq[c], label=c)
    ax.set_yscale("log")
    ax.set_title("Честный walk-forward: рост $1 (лог. шкала, с комиссией)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/equity.png", dpi=120)
    print(f"\nСохранено в {OUT_DIR}/ (summary.csv, yearly.csv, daily_returns.csv, equity.png, dash/*.json)")


if __name__ == "__main__":
    main()
