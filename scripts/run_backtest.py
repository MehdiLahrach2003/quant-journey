import sys
import os
import matplotlib.pyplot as plt

# === Importation des modules internes ===
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# === Fonction d'affichage du backtest ===
def plot_backtest(df, positions, result):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    
    # === 1️⃣ Graphique : Prix et signaux ===
    ax1.plot(df.index, df["price"], label="Prix", color="black", lw=1.2)
    
    # Points d'achat / vente
    buy_signals = positions[(positions.shift(1) <= 0) & (positions > 0)].index
    sell_signals = positions[(positions.shift(1) >= 0) & (positions < 0)].index
    ax1.scatter(buy_signals, df.loc[buy_signals, "price"], color="green", marker="^", label="Buy", s=80)
    ax1.scatter(sell_signals, df.loc[sell_signals, "price"], color="red", marker="v", label="Sell", s=80)
    
    ax1.set_title("SMA Crossover — Signal Trading", fontsize=13)
    ax1.set_ylabel("Prix")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # === 2️⃣ Graphique : Courbe d'équité ===
    ax2.plot(result.equity.index, result.equity, color="blue", lw=1.5, label="Equity Curve")
    ax2.fill_between(result.equity.index, result.equity, result.equity.cummax(), color="red", alpha=0.1)
    ax2.set_title("Évolution du capital (Equity Curve)", fontsize=13)
    ax2.set_ylabel("Capital")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Sauvegarde automatique du graphique
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backtest_plot.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Graphique sauvegardé dans : {save_path}\n")

    plt.show()


# === Fonction principale ===
def main():
    df = load_prices()

    positions = sma_crossover_positions(df["price"], short_window=20, long_window=100)
    result = run_backtest(df["price"], positions, trans_cost_bps=1.0)

    print("\n===== Résultats du Backtest =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")

    print("\n📈 Equity finale :", round(result.equity.iloc[-1], 2))

    # Visualisation
    plot_backtest(df, positions, result)


# === Point d'entrée ===
if __name__ == "__main__":
    main()