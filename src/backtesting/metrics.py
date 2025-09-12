"""
metrics.py

Módulo de cálculo de métricas de performance para estratégias quantitativas e HFT.
Inclui Sharpe Ratio, Sortino, Calmar, retorno acumulado, drawdown, VaR e outros.

Autor: Gabriel Demetrios Lafis
Data: setembro/2025
"""

import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, freq: str = "D") -> float:
    """
    Calcula o Sharpe Ratio anualizado.

    Args:
        returns (pd.Series): Série de retornos percentuais (ex: [0.01, -0.002, ...]).
        risk_free (float): Taxa livre de risco anual (0.0 para simulações).
        freq (str): Frequência dos dados ('D', 'M', 'H').

    Returns:
        float: Sharpe Ratio anualizado.
    """
    freq_map = {"D": 252, "M": 12, "H": 24*252}
    factor = freq_map.get(freq.upper(), 252)
    excess = returns - (risk_free/factor)
    return np.sqrt(factor) * excess.mean() / (excess.std() + 1e-9)

def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, freq: str = "D") -> float:
    """
    Calcula o Sortino Ratio anualizado (penalizando só a volatilidade negativa).
    """
    freq_map = {"D": 252, "M": 12, "H": 24*252}
    factor = freq_map.get(freq.upper(), 252)
    downside = returns[returns < 0]
    divisor = (downside.std() + 1e-9) if not downside.empty else 1
    return np.sqrt(factor) * (returns.mean() - risk_free/factor) / divisor

def calmar_ratio(returns: pd.Series) -> float:
    """
    Calcula o Calmar Ratio: retorno anualizado / drawdown máximo.
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = abs(dd.min())
    ann_return = (cum.iloc[-1])**(252/len(returns)) - 1
    return ann_return / (max_dd + 1e-9)

def max_drawdown(returns: pd.Series) -> float:
    """
    Calcula o drawdown máximo da curva de capital.
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return abs(dd.min())

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calcula o VaR paramétrico em nível de confiança.
    """
    return np.percentile(returns, (1 - confidence) * 100)

def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calcula o CVaR (Expected Shortfall) — média dos piores casos além do VaR.
    """
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def performance_report(returns: pd.Series, risk_free: float = 0.0) -> dict:
    """
    Retorna todas as métricas principais de performance (dict).
    """
    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free),
        "sortino_ratio": sortino_ratio(returns, risk_free),
        "calmar_ratio": calmar_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "total_return": (1 + returns).prod() - 1
    }
