"""
engine.py

Engine principal para backtesting de estratégias quantitativas de alta frequência.
- Suporte a múltiplas estratégias, capital, comissionamento e cálculo automatizado de resultados.
- Estruturado para integração fácil com métricas customizadas e visualização.

Autor: Gabriel Demetrios Lafis
Data: setembro/2025
"""

from typing import List, Dict, Callable, Optional
import pandas as pd

class BacktestEngine:
    """
    Engine de backtesting para estratégias quantitativas.

    - Capaz de rodar múltiplas estratégias sobre dados históricos simulados.
    - Expõe resultados finais e métricas de desempenho.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 1_000_000,
                 commission: float = 0.001,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Inicializa o objeto de backtest.

        Args:
            data (pd.DataFrame): Dados históricos indexados por datetime.
            initial_capital (float): Capital inicial para simulação.
            commission (float): Percentual de comissão por ordem executada.
            start_date (str, opcional): Data inicial.
            end_date (str, opcional): Data final.
        """
        self.data = data
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.start_date = pd.to_datetime(start_date) if start_date else data.index.min()
        self.end_date = pd.to_datetime(end_date) if end_date else data.index.max()
        self.strategy_callbacks: List[Callable] = []
        self.trades: List[Dict] = []
        self.performance: Dict = {}

    def add_strategy(self, strategy):
        """Adiciona uma função de estratégia (callable) para execução."""
        self.strategy_callbacks.append(strategy)

    def run(self):
        """Executa o backtest."""
        current_capital = self.capital
        for dt, row in self.data.loc[self.start_date:self.end_date].iterrows():
            for strat in self.strategy_callbacks:
                orders = strat(row)
                for o in orders:
                    trade_cost = o['quantity'] * row['price'] + abs(o['quantity'] * row['price'] * self.commission)
                    if o['side'] == 'buy':
                        current_capital -= trade_cost
                    elif o['side'] == 'sell':
                        current_capital += trade_cost
                    self.trades.append({'date': dt, **o, 'price': row['price']})
        self.capital = current_capital
        self._calculate_performance()
        return self.performance

    def _calculate_performance(self):
        """Calcula métricas simplificadas de performance."""
        total_return = (self.capital / self.initial_capital - 1) * 100
        n_trades = len(self.trades)
        self.performance = {
            'total_return_percent': total_return,
            'final_capital': self.capital,
            'trades': n_trades
        }
