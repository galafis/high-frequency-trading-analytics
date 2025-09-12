"""
visualization.py

Ferramentas para visualização gráfica de performance em backtesting quantitativo.
Inclui gráficos de:
- Patrimônio líquido (PL) ao longo do tempo
- Drawdown
- Distribuição de retornos e trades
- Relatório consolidado

Autor: Gabriel Demetrios Lafis
Data: setembro/2025
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_equity_curve(returns: pd.Series, title: str = "Equity Curve") -> None:
    """
    Plota a curva de patrimônio líquido acumulado.

    Args:
        returns (pd.Series): Série de retornos diários/horários, etc.
        title (str): Título do gráfico.
    """
    equity = (1 + returns).cumprod()
    plt.figure(figsize=(10, 4))
    equity.plot(label="PL acumulado", color="blue")
    plt.title(title)
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_drawdown(returns: pd.Series, title: str = "Drawdown") -> None:
    """
    Plota o gráfico de drawdown ao longo do tempo.

    Args:
        returns (pd.Series): Série de retornos.
        title (str): Título do gráfico.
    """
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    plt.figure(figsize=(10, 4))
    drawdown.plot(label="Drawdown", color="red")
    plt.title(title)
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_return_histogram(returns: pd.Series, bins: int = 30, title: str = "Distribuição dos Retornos") -> None:
    """
    Plota histograma da distribuição dos retornos.

    Args:
        returns (pd.Series): Retornos.
        bins (int): Número de bins do histograma.
        title (str): Título.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=bins, alpha=0.7, color='gray')
    plt.title(title)
    plt.xlabel("Retorno (%)")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.show()

def plot_performance_summary(returns: pd.Series) -> None:
    """
    Exibe de forma consolidada: equity curve, drawdown e histograma de retornos.
    """
    plt.figure(figsize=(12, 8))
    # PL
    plt.subplot(3, 1, 1)
    equity = (1 + returns).cumprod()
    plt.plot(equity, label="Equity", color='blue')
    plt.title("Curva de Patrimônio Líquido (Equity)")
    plt.legend(); plt.grid(True)
    # Drawdown
    plt.subplot(3, 1, 2)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    plt.plot(drawdown, label="Drawdown", color='red')
    plt.title("Drawdown")
    plt.legend(); plt.grid(True)
    # Histograma
    plt.subplot(3, 1, 3)
    plt.hist(returns, bins=30, alpha=0.7, color='gray')
    plt.title("Distribuição dos Retornos")
    plt.xlabel("Retorno (%)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()
