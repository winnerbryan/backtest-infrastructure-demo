"""
Demo Backtest Infrastructure
Author: Bryan Liew
Date: 2026-03
Note: This script is designed to demonstrate data engineering, vectorized backtesting 
using Polars, and reporting capabilities. The alpha factor implemented here 
(normalized daily candle body) is strictly a toy factor for demonstration purposes.
"""

import ccxt
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from tqdm import tqdm

# ==========================================
# Config
# ==========================================
getcontext().prec = 28
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2025-01-01 00:00:00'
INITIAL_CAPITAL = Decimal('100000.00')
LEVERAGE = Decimal('1.0')
TAKER_FEE = Decimal('0.0005') 
ALPHA_THRESHOLD = 0

# ==========================================
# Data Fetching
# ==========================================
def fetch_data(symbol, timeframe, start_str, end_str):
    print(f"[INFO] Fetching historical data for {symbol}...")
    exchange = ccxt.bybit()
    curr_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    all_ohlcv = []
    
    with tqdm(desc="Downloading", unit="k-lines") as pbar:
        while curr_ts < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=curr_ts, limit=1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                curr_ts = ohlcv[-1][0] + 1
                pbar.update(len(ohlcv) // 1000)
            except Exception as e:
                print(f"[ERROR] Fetch failed: {e}")
                break
    
    if not all_ohlcv: 
        raise ValueError("No data fetched.")
    
    df = pl.DataFrame(all_ohlcv, schema=['ts', 'open', 'high', 'low', 'close', 'vol'])
    return df.with_columns(pl.from_epoch("ts", time_unit="ms"))

# ==========================================
# Backtest Engine
# ==========================================
def run_backtest(df: pl.DataFrame):
    print("[INFO] Running backtest engine...")
    
    df = df.with_columns([
        ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 0.001)).alias("alpha")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("alpha") > ALPHA_THRESHOLD).then(1)
          .when(pl.col("alpha") < -ALPHA_THRESHOLD).then(-1)
          .otherwise(0).alias("signal")
    ]).with_columns(pl.col("signal").shift(1).fill_null(0).alias("position"))

    prices = df['close'].to_list()
    positions = df['position'].to_list()
    
    equity = INITIAL_CAPITAL
    gross_equity = INITIAL_CAPITAL
    benchmark_equity = INITIAL_CAPITAL
    
    equity_curve = [float(INITIAL_CAPITAL)]
    gross_curve = [float(INITIAL_CAPITAL)]
    benchmark_curve = [float(INITIAL_CAPITAL)]
    
    trade_records = []
    entry_equity = equity
    last_pos = 0
    
    for i in range(1, len(prices)):
        p_curr = Decimal(str(prices[i]))
        p_prev = Decimal(str(prices[i-1]))
        curr_pos = positions[i]
        
        benchmark_equity = benchmark_equity * (p_curr / p_prev)
        
        if last_pos != 0:
            pnl_ratio = (p_curr / p_prev - 1) * Decimal(str(last_pos)) * LEVERAGE
            equity = equity * (1 + pnl_ratio)
            gross_equity = gross_equity * (1 + pnl_ratio)
        
        if curr_pos != last_pos:
            fee = equity * TAKER_FEE * LEVERAGE
            equity -= fee
            
            if last_pos != 0:
                trade_ret = float(equity / entry_equity) - 1
                trade_records.append({'pnl': trade_ret, 'side': last_pos})
                
            entry_equity = equity
            
        equity_curve.append(float(equity))
        gross_curve.append(float(gross_equity))
        benchmark_curve.append(float(benchmark_equity))
        last_pos = curr_pos

    res_df = df.with_columns([
        pl.Series("equity", equity_curve),
        pl.Series("gross_equity", gross_curve),
        pl.Series("benchmark", benchmark_curve)
    ])
    return res_df, trade_records

# ==========================================
# Metrics Calculation
# ==========================================
def calc_metrics(res_df, trade_records):
    trades_pnl = [t['pnl'] for t in trade_records]
    equity_series = res_df['equity'].to_numpy()
    returns = np.nan_to_num(np.diff(equity_series) / equity_series[:-1])
    
    final_equity = equity_series[-1]
    total_return = (final_equity / float(INITIAL_CAPITAL)) - 1
    bench_return = (res_df['benchmark'][-1] / float(INITIAL_CAPITAL)) - 1
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    downside_returns = returns[returns < 0]
    sortino = mean_ret / np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    pos_pnl = sum([p for p in trades_pnl if p > 0])
    neg_pnl = sum([p for p in trades_pnl if p < 0])
    pf = pos_pnl / abs(neg_pnl) if abs(neg_pnl) > 0 else 0
    ev = np.mean(trades_pnl) if trades_pnl else 0
    
    peak = pl.Series(equity_series).cum_max()
    dd = (pl.Series(equity_series) - peak) / peak
    
    wins = [p for p in trades_pnl if p > 0]
    win_rate = len(wins) / len(trades_pnl) if trades_pnl else 0

    long_trades = [t for t in trade_records if t['side'] == 1]
    short_trades = [t for t in trade_records if t['side'] == -1]
    
    long_win_rate = len([t for t in long_trades if t['pnl'] > 0]) / len(long_trades) if long_trades else 0
    short_win_rate = len([t for t in short_trades if t['pnl'] > 0]) / len(short_trades) if short_trades else 0

    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for pnl in trades_pnl:
        if pnl > 0:
            cur_win += 1; cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        elif pnl < 0:
            cur_loss += 1; cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
            
    gross_final = res_df['gross_equity'][-1]
    fee_impact = (gross_final - final_equity) / gross_final if gross_final > 0 else 0

    return {
        "final_equity": final_equity, "gross_equity": gross_final, "fee_impact": fee_impact,
        "total_return": total_return, "bench_return": bench_return,
        "sharpe": sharpe, "sortino": sortino, "pf": pf, "ev": ev,
        "max_dd": dd.min(), "trades": len(trades_pnl), "dd_series": dd,
        "win_rate": win_rate, "max_win_streak": max_win_streak, "max_loss_streak": max_loss_streak,
        "long_count": len(long_trades), "short_count": len(short_trades),
        "long_win_rate": long_win_rate, "short_win_rate": short_win_rate
    }

# ==========================================
# Reporting (Side-by-Side Tables with Color Coding)
# ==========================================
def get_color(val, threshold=0.0):
    """根据数值大小动态返回颜色：绿涨红跌"""
    if val > threshold: return '#00ff00' # Green
    elif val < threshold: return '#ff4444' # Red
    return 'white'

def generate_report(df, metrics, symbol):
    print("[INFO] Generating performance tear sheet...")
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1, 1.8], width_ratios=[1, 1]) 
    
    dates, equity, gross, benchmark = df['ts'].to_numpy(), df['equity'].to_numpy(), df['gross_equity'].to_numpy(), df['benchmark'].to_numpy()
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, gross, label='Gross Equity', color='cyan', linestyle=':', linewidth=1.5, alpha=0.8)
    ax1.plot(dates, equity, label='Net Equity', color='#00ff00', linewidth=2.5)
    ax1.plot(dates, benchmark, label='Benchmark', color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_title(f"Demo Backtest Tear Sheet - {symbol}", fontsize=22, fontweight='bold', color='white', pad=20)
    ax1.set_ylabel("Equity (USDT)", fontsize=14)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.fill_between(dates, metrics['dd_series'].to_numpy(), 0, color='#ff4444', alpha=0.6)
    ax2.set_ylabel("Drawdown %", fontsize=14)
    ax2.grid(True, alpha=0.2)
    
    # ==========================================
    # 数据定义与颜色归类 (Label, Value, ValueColor)
    # ==========================================
    
    # 左侧：核心指标 (移入了 Benchmark 和 Sortino)
    table1_rows = [
        ("Initial Capital", f"${float(INITIAL_CAPITAL):,.2f}", 'white'),
        ("Final Net Equity", f"${metrics['final_equity']:,.2f}", 'cyan'),
        ("Total Net Return", f"{metrics['total_return']*100:.2f}%", get_color(metrics['total_return'])),
        ("Benchmark Return", f"{metrics['bench_return']*100:.2f}%", get_color(metrics['bench_return'])),
        ("Max Drawdown", f"{metrics['max_dd']*100:.2f}%", '#ff4444'),
        ("Sharpe Ratio", f"{metrics['sharpe']:.4f}", get_color(metrics['sharpe'])),
        ("Sortino Ratio", f"{metrics['sortino']:.4f}", get_color(metrics['sortino'])),
        ("Expected Value (EV)", f"{metrics['ev']*100:.3f}%", get_color(metrics['ev'])),
        ("Profit Factor", f"{metrics['pf']:.2f}", get_color(metrics['pf'], threshold=1.0)),
        ("Total Trades", f"{metrics['trades']}", 'white'),
        ("Total Win Rate", f"{metrics['win_rate']*100:.2f}%", get_color(metrics['win_rate'], threshold=0.5))
    ]
    
    # 右侧：多空与附属分析
    table2_rows = [
        ("Long Trades", f"{metrics['long_count']}", '#00ff00'),       # 多头一律用绿色
        ("Long Win Rate", f"{metrics['long_win_rate']*100:.2f}%", '#00ff00'),
        ("Short Trades", f"{metrics['short_count']}", '#ff9900'),      # 空头一律用橙色
        ("Short Win Rate", f"{metrics['short_win_rate']*100:.2f}%", '#ff9900'),
        ("Max Win / Loss Streak", f"{metrics['max_win_streak']} / {metrics['max_loss_streak']}", 'white'),
        ("Final Gross Equity", f"${metrics['gross_equity']:,.2f}", 'cyan'),
        ("Fee Impact", f"-{metrics['fee_impact']*100:.2f}%", '#ff4444')
    ]

    # ==========================================
    # 表格绘制逻辑
    # ==========================================
    def draw_table(ax, title, title_color, rows):
        ax.axis('off')
        # 组装数据
        cell_text = [[title, "VALUE"]] + [[r[0], r[1]] for r in rows]
        table = ax.table(cellText=cell_text, loc='center', cellLoc='center', colWidths=[0.55, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2.4)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(1)
            cell.set_edgecolor('#444444')
            if row == 0: # 表头
                cell.set_facecolor('#2b2b2b')
                cell.set_text_props(color=title_color if col == 0 else 'white', weight='bold')
            else: # 内容
                cell.set_facecolor('#111111')
                if col == 0: # Label列 (统一灰白色)
                    cell.set_text_props(color='#cccccc')
                elif col == 1: # Value列 (应用专属颜色逻辑)
                    cell.set_text_props(color=rows[row-1][2], weight='bold')

    draw_table(fig.add_subplot(gs[2, 0]), "CORE METRICS", "#00ffcc", table1_rows)
    draw_table(fig.add_subplot(gs[2, 1]), "LONG / SHORT ANALYSIS", "#ff9900", table2_rows)

    plt.tight_layout()
    plt.savefig('tearsheet_pro.png', dpi=150, bbox_inches='tight')
    print("[INFO] Report saved as tearsheet_pro.png")

if __name__ == "__main__":
    try:
        raw_df = fetch_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
        res_df, trades = run_backtest(raw_df)
        
        if not trades:
            print("[WARN] No trades executed.")
        else:
            metrics = calc_metrics(res_df, trades)
            generate_report(res_df, metrics, SYMBOL)
            
    except Exception as e:
        print(f"[ERROR] Engine halted: {e}")