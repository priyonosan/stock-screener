#!/usr/bin/env python3
"""
Backtesting Engine - Minervini Trend Template
=============================================
Simulasi sinyal beli/jual historis 3 tahun kebelakang
menggunakan logika yang sama dengan run_optimized_scan.py

Usage:
    python run_backtest.py
    python run_backtest.py --years 3
    python run_backtest.py --use-config
    python run_backtest.py --capital 10000
"""

import argparse
import logging
import sys
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  PHASE CLASSIFIER (sama dengan screener)
# ─────────────────────────────────────────────
def classify_phase(closes: pd.Series) -> int:
    if len(closes) < 200:
        return 0
    sma50  = closes.rolling(50).mean()
    sma150 = closes.rolling(150).mean()
    sma200 = closes.rolling(200).mean()

    s50  = sma50.iloc[-1]
    s150 = sma150.iloc[-1]
    s200 = sma200.iloc[-1]
    price = closes.iloc[-1]

    slope200 = (sma200.iloc[-1] - sma200.iloc[-21]) / sma200.iloc[-21] if sma200.iloc[-21] > 0 else 0

    if (s50 > s150 > s200 and slope200 > 0 and price > s50):
        return 2  # Uptrend
    elif (s50 < s200 and slope200 < 0 and price < s50):
        return 4  # Downtrend
    elif (s50 > s200 and price < s50):
        return 3  # Distribution
    else:
        return 1  # Base building


def minervini_score(closes: pd.Series, spy_closes: pd.Series) -> float:
    """Skor sederhana 0-100 berdasarkan Minervini Trend Template."""
    if len(closes) < 200:
        return 0

    try:
        sma200_series = closes.rolling(200).mean()

        # Pastikan semua nilai scalar (float), bukan Series
        price      = float(closes.iloc[-1])
        sma50      = float(closes.rolling(50).mean().iloc[-1])
        sma150     = float(closes.rolling(150).mean().iloc[-1])
        sma200     = float(sma200_series.iloc[-1])
        sma200_21d = float(sma200_series.iloc[-21]) if len(sma200_series) >= 21 else sma200
        high52     = float(closes.rolling(min(252, len(closes))).max().iloc[-1])
        low52      = float(closes.rolling(min(252, len(closes))).min().iloc[-1])

        # Validasi tidak ada NaN
        import math
        if any(math.isnan(v) for v in [price, sma50, sma150, sma200]):
            return 0

        score = 0

        # Kriteria 1: Price > SMA150 dan SMA200
        if price > sma150 and price > sma200:
            score += 20

        # Kriteria 2: SMA150 > SMA200
        if sma150 > sma200:
            score += 15

        # Kriteria 3: SMA200 slope positif
        slope200 = (sma200 - sma200_21d) / sma200_21d if sma200_21d > 0 else 0.0
        if slope200 > 0:
            score += 15

        # Kriteria 4: SMA50 > SMA150 > SMA200
        if sma50 > sma150 > sma200:
            score += 20

        # Kriteria 5: Price > SMA50
        if price > sma50:
            score += 10

        # Kriteria 6-7: jarak dari high/low 52 minggu
        if low52 > 0 and (price / low52 - 1) >= 0.30:
            score += 10
        if high52 > 0 and (price / high52) >= 0.75:
            score += 10

        return score

    except Exception:
        return 0


# ─────────────────────────────────────────────
#  DOWNLOAD DATA
# ─────────────────────────────────────────────
def download_data(tickers: list, years: int) -> dict:
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 60)  # buffer extra

    logger.info(f"Downloading {len(tickers)} tickers | {start.date()} → {end.date()}")

    all_data = {}
    failed   = []

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False, timeout=15)
            if df is not None and len(df) >= 200:
                all_data[ticker] = df['Close'].dropna()
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)

        if (i + 1) % 50 == 0:
            logger.info(f"  Downloaded {i+1}/{len(tickers)}...")

    logger.info(f"Success: {len(all_data)} | Failed: {len(failed)}")
    if failed:
        logger.info(f"Failed tickers: {', '.join(failed[:20])}{'...' if len(failed)>20 else ''}")

    return all_data


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(all_data: dict, spy_data: pd.Series,
                 capital: float, score_threshold: int) -> dict:
    """
    Simulasi walk-forward:
    - Setiap hari scan seluruh saham
    - Beli jika score >= threshold dan belum punya posisi
    - Jual jika phase berubah ke 3/4 atau stop loss tersentuh
    """

    trades       = []
    equity_curve = []
    cash         = capital
    positions    = {}   # {ticker: {'entry': price, 'date': date, 'stop': price, 'shares': n}}

    # Kumpulkan semua tanggal trading
    all_dates = sorted(set(
        date for closes in all_data.values() for date in closes.index
    ))

    max_positions = 10
    risk_per_trade = 0.02  # 2% risiko per trade

    for date in all_dates:
        equity = cash + sum(
            pos['shares'] * all_data[t].get(date, all_data[t].iloc[-1])
            for t, pos in positions.items()
            if t in all_data
        )
        equity_curve.append({'date': date, 'equity': equity})

        # ── CEK EXIT dulu ──
        to_exit = []
        for ticker, pos in positions.items():
            if ticker not in all_data:
                continue
            closes = all_data[ticker]
            idx    = closes.index.get_indexer([date], method='nearest')[0]
            if idx < 0:
                continue
            current_price = closes.iloc[idx]
            window        = closes.iloc[max(0, idx-200):idx+1]
            phase         = classify_phase(window)

            hit_stop = current_price <= pos['stop']
            bad_phase = phase in [3, 4]
            held_days = (date - pos['date']).days

            if hit_stop or bad_phase or held_days > 180:
                reason = 'stop_loss' if hit_stop else ('phase_exit' if bad_phase else 'time_exit')
                pnl    = (current_price - pos['entry']) * pos['shares']
                ret_pct = (current_price / pos['entry'] - 1) * 100
                cash   += current_price * pos['shares']
                trades.append({
                    'ticker':     ticker,
                    'entry_date': pos['date'],
                    'exit_date':  date,
                    'entry':      pos['entry'],
                    'exit':       current_price,
                    'shares':     pos['shares'],
                    'pnl':        pnl,
                    'return_pct': ret_pct,
                    'days_held':  held_days,
                    'reason':     reason,
                })
                to_exit.append(ticker)

        for t in to_exit:
            del positions[t]

        # ── SCAN BUY ──
        if len(positions) < max_positions:
            candidates = []
            for ticker, closes in all_data.items():
                if ticker in positions:
                    continue
                idx = closes.index.get_indexer([date], method='nearest')[0]
                if idx < 200:
                    continue
                window = closes.iloc[idx-200:idx+1]
                spy_w  = spy_data.reindex(window.index, method='nearest')
                score  = minervini_score(window, spy_w)
                if score >= score_threshold:
                    candidates.append((ticker, score, closes.iloc[idx]))

            candidates.sort(key=lambda x: x[1], reverse=True)

            slots = max_positions - len(positions)
            for ticker, score, price in candidates[:slots]:
                atr   = pd.Series([
                    closes.iloc[max(0,i-1):i+1].max() - closes.iloc[max(0,i-1):i+1].min()
                    for i in range(max(1, len(all_data[ticker])-14), len(all_data[ticker]))
                ]).mean()
                stop  = price - (atr * 2.0)
                risk  = price - stop
                if risk <= 0:
                    continue
                position_size = (capital * risk_per_trade) / risk
                cost          = position_size * price
                if cost > cash:
                    position_size = cash / price
                    cost          = cash
                if position_size < 1:
                    continue

                shares = int(position_size)
                cash  -= shares * price
                positions[ticker] = {
                    'entry':  price,
                    'date':   date,
                    'stop':   stop,
                    'shares': shares,
                    'score':  score,
                }

    # Tutup posisi yang masih buka di akhir
    last_date = all_dates[-1]
    for ticker, pos in positions.items():
        if ticker not in all_data:
            continue
        current_price = all_data[ticker].iloc[-1]
        pnl           = (current_price - pos['entry']) * pos['shares']
        ret_pct       = (current_price / pos['entry'] - 1) * 100
        held_days     = (last_date - pos['date']).days
        cash         += current_price * pos['shares']
        trades.append({
            'ticker':     ticker,
            'entry_date': pos['date'],
            'exit_date':  last_date,
            'entry':      pos['entry'],
            'exit':       current_price,
            'shares':     pos['shares'],
            'pnl':        pnl,
            'return_pct': ret_pct,
            'days_held':  held_days,
            'reason':     'still_open',
        })

    return {
        'trades':       trades,
        'equity_curve': equity_curve,
        'final_equity': cash,
    }


# ─────────────────────────────────────────────
#  HITUNG STATISTIK
# ─────────────────────────────────────────────
def compute_stats(results: dict, capital: float,
                  spy_data: pd.Series, years: int) -> dict:
    trades = results['trades']
    equity = pd.DataFrame(results['equity_curve']).set_index('date')['equity']

    if not trades:
        return {'error': 'Tidak ada trade yang terjadi'}

    df     = pd.DataFrame(trades)
    closed = df[df['reason'] != 'still_open']
    wins   = closed[closed['pnl'] > 0]
    losses = closed[closed['pnl'] <= 0]

    # ── Durasi aktual dari data (lebih akurat daripada pakai 'years') ──
    actual_days  = (equity.index[-1] - equity.index[0]).days
    actual_years = actual_days / 365.25

    final_equity = results['final_equity']

    # ── Trade stats ──
    win_rate      = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win       = wins['return_pct'].mean()    if len(wins) > 0 else 0
    avg_loss      = losses['return_pct'].mean()  if len(losses) > 0 else 0
    gross_profit  = wins['pnl'].sum()            if len(wins) > 0 else 0
    gross_loss    = abs(losses['pnl'].sum())     if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss    if gross_loss > 0 else float('inf')
    expectancy    = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # ── Return metrics ──
    total_return  = (final_equity / capital - 1) * 100

    # CAGR formula yang benar: (End/Start)^(1/years) - 1
    cagr = ((final_equity / capital) ** (1 / actual_years) - 1) * 100 if actual_years > 0 else 0

    # SPY CAGR untuk perbandingan
    spy_total_return = (spy_data.iloc[-1] / spy_data.iloc[0] - 1) * 100
    spy_cagr = ((spy_data.iloc[-1] / spy_data.iloc[0]) ** (1 / actual_years) - 1) * 100 if actual_years > 0 else 0

    # ── Drawdown ──
    peak         = equity.cummax()
    dd           = (equity - peak) / peak * 100
    max_drawdown = dd.min()
    # Calmar ratio: CAGR / Max Drawdown (makin tinggi makin baik)
    calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')

    # Rata-rata drawdown
    avg_drawdown = dd[dd < 0].mean() if len(dd[dd < 0]) > 0 else 0

    # Durasi max drawdown
    in_drawdown     = dd < 0
    dd_start        = None
    max_dd_duration = 0
    current_dur     = 0
    for val in in_drawdown:
        if val:
            current_dur += 1
            max_dd_duration = max(max_dd_duration, current_dur)
        else:
            current_dur = 0

    # ── Sharpe & Sortino ratio ──
    daily_returns  = equity.pct_change().dropna()
    rf_daily       = 0.05 / 252  # risk-free rate 5% per tahun
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    downside = daily_returns[daily_returns < rf_daily]
    sortino  = (excess_returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

    # ── Trade frequency ──
    trades_per_year = len(closed) / actual_years if actual_years > 0 else 0

    # ── Exit reason breakdown ──
    reason_counts = closed['reason'].value_counts().to_dict() if len(closed) > 0 else {}

    # ── Modal akhir ──
    final_value = capital + (final_equity - capital)

    return {
        # Trade stats
        'total_trades':      len(closed),
        'open_trades':       len(df[df['reason'] == 'still_open']),
        'winning_trades':    len(wins),
        'losing_trades':     len(losses),
        'win_rate':          win_rate,
        'avg_win_pct':       avg_win,
        'avg_loss_pct':      avg_loss,
        'profit_factor':     profit_factor,
        'expectancy':        expectancy,
        'trades_per_year':   trades_per_year,
        'avg_days_held':     closed['days_held'].mean() if len(closed) > 0 else 0,
        'gross_profit':      gross_profit,
        'gross_loss':        gross_loss,
        'net_profit':        gross_profit - gross_loss,
        'reason_counts':     reason_counts,

        # Return metrics
        'capital':           capital,
        'final_value':       final_equity,
        'total_return':      total_return,
        'cagr':              cagr,                  # ← CAGR utama
        'actual_years':      actual_years,
        'spy_total_return':  spy_total_return,
        'spy_cagr':          spy_cagr,              # ← CAGR SPY untuk perbandingan
        'alpha_total':       total_return - spy_total_return,
        'alpha_cagr':        cagr - spy_cagr,       # ← Alpha CAGR

        # Risk metrics
        'max_drawdown':      max_drawdown,
        'avg_drawdown':      avg_drawdown,
        'max_dd_duration':   max_dd_duration,
        'sharpe_ratio':      sharpe,
        'sortino_ratio':     sortino,
        'calmar_ratio':      calmar_ratio,          # ← CAGR / Max DD

        # Best/Worst
        'best_trade':        df.loc[df['return_pct'].idxmax()][['ticker','return_pct','days_held','entry_date','exit_date']].to_dict() if len(df) > 0 else {},
        'worst_trade':       df.loc[df['return_pct'].idxmin()][['ticker','return_pct','days_held','entry_date','exit_date']].to_dict() if len(df) > 0 else {},
        'trades_df':         df,
    }


# ─────────────────────────────────────────────
#  SIMPAN LAPORAN
# ─────────────────────────────────────────────
def save_report(stats: dict, capital: float, years: int,
                output_dir: str = "./data/backtests"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    lines = []
    lines.append("=" * 80)
    lines.append("BACKTEST REPORT - MINERVINI TREND TEMPLATE")
    lines.append(f"Periode    : {years} tahun kebelakang ({stats.get('actual_years', years):.2f} tahun aktual)")
    lines.append(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Modal Awal : ${capital:,.0f}")
    if 'final_value' in stats:
        lines.append(f"Modal Akhir: ${stats['final_value']:,.0f}  (Net P&L: ${stats['final_value']-capital:+,.0f})")
    lines.append("=" * 80)
    lines.append("")

    if 'error' in stats:
        lines.append(f"ERROR: {stats['error']}")
    else:
        # ── CAGR SECTION (utama) ──────────────────────────────────────
        lines.append("╔══ CAGR & RETURN SUMMARY ═══════════════════════════════════════╗")
        lines.append(f"║  CAGR Strategi          : {stats['cagr']:>+8.2f}%  per tahun        ║")
        lines.append(f"║  CAGR SPY (benchmark)   : {stats['spy_cagr']:>+8.2f}%  per tahun        ║")
        lines.append(f"║  Alpha CAGR vs SPY      : {stats['alpha_cagr']:>+8.2f}%  per tahun        ║")
        lines.append(f"║  ─────────────────────────────────────────────────────────── ║")
        lines.append(f"║  Total Return Strategi  : {stats['total_return']:>+8.1f}%                   ║")
        lines.append(f"║  Total Return SPY       : {stats['spy_total_return']:>+8.1f}%                   ║")
        lines.append(f"║  Alpha Total            : {stats['alpha_total']:>+8.1f}%                   ║")
        lines.append("╚═════════════════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append("  Formula CAGR: (Modal Akhir / Modal Awal)^(1/Tahun) - 1")
        lines.append(f"  = (${stats['final_value']:,.0f} / ${capital:,.0f})^(1/{stats['actual_years']:.2f}) - 1 = {stats['cagr']:+.2f}%")
        lines.append("")

        # ── TRADE STATS ───────────────────────────────────────────────
        lines.append("── STATISTIK TRADE ─────────────────────────────────────────────")
        lines.append(f"  Total Trade (closed)   : {stats['total_trades']}")
        lines.append(f"  Trade masih buka       : {stats['open_trades']}")
        lines.append(f"  Winning trades         : {stats['winning_trades']}  ({stats['win_rate']:.1f}%)")
        lines.append(f"  Losing trades          : {stats['losing_trades']}  ({100-stats['win_rate']:.1f}%)")
        lines.append(f"  Trade per tahun        : {stats['trades_per_year']:.1f}")
        lines.append(f"  Avg hari dipegang      : {stats['avg_days_held']:.0f} hari")
        lines.append("")
        lines.append(f"  Avg Win                : +{stats['avg_win_pct']:.1f}%")
        lines.append(f"  Avg Loss               : {stats['avg_loss_pct']:.1f}%")
        lines.append(f"  Profit Factor          : {stats['profit_factor']:.2f}x  (>1.5 = bagus, >2.0 = sangat bagus)")
        lines.append(f"  Expectancy per trade   : {stats['expectancy']:+.2f}%")
        lines.append(f"  Gross Profit           : ${stats['gross_profit']:,.0f}")
        lines.append(f"  Gross Loss             : ${stats['gross_loss']:,.0f}")
        lines.append(f"  Net Profit             : ${stats['net_profit']:+,.0f}")
        lines.append("")

        # ── EXIT REASONS ─────────────────────────────────────────────
        if stats.get('reason_counts'):
            lines.append("  Exit breakdown:")
            for reason, count in stats['reason_counts'].items():
                pct = count / stats['total_trades'] * 100
                lines.append(f"    {reason:<15}: {count:>4} trade ({pct:.1f}%)")
        lines.append("")

        # ── RISK METRICS ─────────────────────────────────────────────
        lines.append("── METRIK RISIKO ───────────────────────────────────────────────")
        lines.append(f"  Max Drawdown           : {stats['max_drawdown']:.1f}%")
        lines.append(f"  Avg Drawdown           : {stats['avg_drawdown']:.1f}%")
        lines.append(f"  Max DD Duration        : {stats['max_dd_duration']} hari")
        lines.append(f"  Sharpe Ratio           : {stats['sharpe_ratio']:.2f}  (>1.0 = bagus, >2.0 = sangat bagus)")
        lines.append(f"  Sortino Ratio          : {stats['sortino_ratio']:.2f}  (fokus downside risk)")
        lines.append(f"  Calmar Ratio           : {stats['calmar_ratio']:.2f}  (CAGR / Max DD, >1.0 = bagus)")
        lines.append("")

        # ── INTERPRETASI ─────────────────────────────────────────────
        lines.append("── INTERPRETASI ────────────────────────────────────────────────")
        cagr = stats['cagr']
        spy_cagr = stats['spy_cagr']
        if cagr > spy_cagr + 5:
            lines.append("  ✅ Strategi OUTPERFORM SPY secara signifikan")
        elif cagr > spy_cagr:
            lines.append("  🟡 Strategi sedikit outperform SPY")
        else:
            lines.append("  ❌ Strategi UNDERPERFORM SPY — pertimbangkan beli SPY saja")

        if stats['profit_factor'] >= 2.0:
            lines.append("  ✅ Profit Factor sangat baik (>=2.0)")
        elif stats['profit_factor'] >= 1.5:
            lines.append("  🟡 Profit Factor cukup baik (>=1.5)")
        else:
            lines.append("  ❌ Profit Factor rendah (<1.5) — perlu optimasi")

        if stats['max_drawdown'] > -30:
            lines.append("  ✅ Max Drawdown terkontrol (<30%)")
        elif stats['max_drawdown'] > -50:
            lines.append("  🟡 Max Drawdown cukup besar (30-50%)")
        else:
            lines.append("  ❌ Max Drawdown sangat besar (>50%) — risiko tinggi")

        if stats['sharpe_ratio'] >= 1.0:
            lines.append("  ✅ Sharpe Ratio baik (>=1.0)")
        else:
            lines.append("  🟡 Sharpe Ratio rendah (<1.0)")
        lines.append("")

        # ── BEST/WORST TRADE ─────────────────────────────────────────
        lines.append("── TRADE TERBAIK & TERBURUK ────────────────────────────────────")
        if stats['best_trade']:
            b = stats['best_trade']
            lines.append(f"  Best  : {b['ticker']} → {b['return_pct']:+.1f}% ({b['days_held']:.0f} hari)")
        if stats['worst_trade']:
            w = stats['worst_trade']
            lines.append(f"  Worst : {w['ticker']} → {w['return_pct']:+.1f}% ({w['days_held']:.0f} hari)")
        lines.append("")

        # Top 20 trades
        lines.append("── TOP 20 TRADE (by PnL) ───────────────────────────────────────")
        df = stats['trades_df'].copy()
        df_sorted = df.sort_values('pnl', ascending=False).head(20)
        lines.append(f"  {'Ticker':<8} {'Entry':>10} {'Exit':>10} {'Return':>8} {'PnL':>10} {'Days':>6} {'Reason':<12}")
        lines.append("  " + "-"*70)
        for _, row in df_sorted.iterrows():
            lines.append(
                f"  {row['ticker']:<8} "
                f"{str(row['entry_date'].date()):>10} "
                f"{str(row['exit_date'].date()):>10} "
                f"{row['return_pct']:>+7.1f}% "
                f"${row['pnl']:>9.0f} "
                f"{row['days_held']:>6.0f} "
                f"{row['reason']:<12}"
            )

        lines.append("")
        lines.append("── BOTTOM 10 TRADE (by PnL) ────────────────────────────────────")
        df_worst = df.sort_values('pnl').head(10)
        for _, row in df_worst.iterrows():
            lines.append(
                f"  {row['ticker']:<8} "
                f"{str(row['entry_date'].date()):>10} "
                f"{str(row['exit_date'].date()):>10} "
                f"{row['return_pct']:>+7.1f}% "
                f"${row['pnl']:>9.0f} "
                f"{row['days_held']:>6.0f} "
                f"{row['reason']:<12}"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("DISCLAIMER: Backtest tidak menjamin hasil di masa depan.")
    lines.append("=" * 80)

    report = "\n".join(lines)
    path   = Path(output_dir) / f"backtest_{timestamp}.txt"
    latest = Path(output_dir) / "latest_backtest.txt"

    with open(path, 'w') as f:
        f.write(report)
    with open(latest, 'w') as f:
        f.write(report)

    # Simpan juga CSV semua trade
    if 'trades_df' in stats:
        csv_path = Path(output_dir) / f"backtest_trades_{timestamp}.csv"
        stats['trades_df'].to_csv(csv_path, index=False)
        logger.info(f"Trade CSV: {csv_path}")

    logger.info(f"Report: {path}")
    print(report)
    return path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Backtest Minervini Trend Template')
    parser.add_argument('--years',        type=int,   default=3,     help='Tahun historis (default: 3)')
    parser.add_argument('--capital',      type=float, default=10000, help='Modal awal USD (default: 10000)')
    parser.add_argument('--score',        type=int,   default=70,    help='Min score beli (default: 70)')
    parser.add_argument('--use-config',   action='store_true',       help='Baca ticker dari config.yaml')
    parser.add_argument('--output-dir',   type=str,   default='./data/backtests')
    args = parser.parse_args()

    # ── Load tickers ──
    if args.use_config:
        logger.info("Loading tickers dari config.yaml...")
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        tickers = config.get("stock_universe", [])
        logger.info(f"Loaded {len(tickers)} tickers dari config.yaml")
    else:
        # Default: top 50 liquid stocks
        tickers = [
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","LLY","AVGO","JPM",
            "V","MA","UNH","XOM","COST","HD","PG","JNJ","ABBV","MRK","AMD","CRM",
            "ACN","TMO","BAC","ORCL","NFLX","CSCO","PEP","WMT","ADBE","TXN","QCOM",
            "DHR","RTX","INTU","AMAT","ISRG","SPGI","NOW","GS","BLK","PANW","KLAC",
            "LRCX","SNPS","CDNS","MU","ADI","MRVL"
        ]
        logger.info(f"Menggunakan {len(tickers)} default tickers")

    # ── Download data ──
    all_data = download_data(tickers, args.years)
    if not all_data:
        logger.error("Gagal download data")
        sys.exit(1)

    # ── Download SPY ──
    logger.info("Downloading SPY benchmark...")
    end   = datetime.today()
    start = end - timedelta(days=365 * args.years + 60)
    spy_df   = yf.download("SPY", start=start, end=end,
                            auto_adjust=True, progress=False)
    spy_data = spy_df['Close'].dropna()

    # ── Jalankan backtest ──
    logger.info(f"Menjalankan backtest {args.years} tahun | Modal: ${args.capital:,.0f} | Min score: {args.score}")
    results = run_backtest(all_data, spy_data, args.capital, args.score)

    # ── Statistik ──
    stats = compute_stats(results, args.capital, spy_data, args.years)

    # ── Simpan laporan ──
    save_report(stats, args.capital, args.years, args.output_dir)


if __name__ == '__main__':
    main()
