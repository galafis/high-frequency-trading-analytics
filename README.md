# High-Frequency Trading Analytics

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Sistema avançado de análise quantitativa e trading de alta frequência com algoritmos de machine learning, reinforcement learning e análise de microestrutura de mercado para estratégias automatizadas.

## 🎯 Visão Geral

Plataforma completa de trading quantitativo que combina análise de dados financeiros em tempo real, modelos preditivos avançados e estratégias de execução automatizada para mercados de alta frequência.

### ✨ Características Principais

- **🤖 Reinforcement Learning**: Agentes DQN, PPO e A3C para trading
- **📊 Análise Quantitativa**: Modelos estatísticos e econométricos
- **⚡ Baixa Latência**: Otimização para execução em microssegundos
- **📈 Backtesting**: Framework robusto de teste histórico
- **🔍 Microestrutura**: Análise de order book e market impact
- **⚖️ Risk Management**: Gestão de risco em tempo real

## 🛠️ Stack Tecnológico

### Machine Learning & AI
- **PyTorch**: Deep learning e reinforcement learning
- **Scikit-learn**: Modelos de machine learning clássico
- **TensorFlow**: Redes neurais alternativas
- **Stable-Baselines3**: Algoritmos RL prontos

### Análise Quantitativa
- **NumPy**: Computação numérica otimizada
- **Pandas**: Manipulação de dados financeiros
- **SciPy**: Análise estatística avançada
- **QuantLib**: Biblioteca de finanças quantitativas

### Dados e Performance
- **Numba**: Compilação JIT para performance
- **Cython**: Extensões C para código crítico
- **Redis**: Cache de dados em tempo real
- **InfluxDB**: Banco de dados de séries temporais

## 📁 Estrutura do Projeto

```
high-frequency-trading-analytics/
├── src/
│   ├── models/                     # Modelos de ML/RL
│   │   ├── reinforcement_learning.py  # Agentes RL (DQN, PPO, A3C)
│   │   ├── predictive_models.py       # Modelos preditivos
│   │   └── risk_models.py             # Modelos de risco
│   ├── strategies/                 # Estratégias de trading
│   │   ├── market_making.py           # Market making
│   │   ├── arbitrage.py               # Arbitragem estatística
│   │   └── momentum.py                # Estratégias de momentum
│   ├── data/                       # Processamento de dados
│   │   ├── market_data.py             # Dados de mercado
│   │   ├── order_book.py              # Order book processing
│   │   └── features.py                # Feature engineering
│   ├── execution/                  # Execução de ordens
│   │   ├── order_manager.py           # Gerenciamento de ordens
│   │   ├── slippage.py                # Análise de slippage
│   │   └── latency.py                 # Otimização de latência
│   ├── backtesting/                # Framework de backtesting
│   │   ├── engine.py                  # Engine principal
│   │   ├── metrics.py                 # Métricas de performance
│   │   └── visualization.py           # Visualizações
│   └── utils/                      # Utilitários
│       ├── config.py                  # Configurações
│       ├── logger.py                  # Sistema de logs
│       └── helpers.py                 # Funções auxiliares
├── notebooks/                      # Jupyter notebooks
├── tests/                          # Testes automatizados
├── data/                           # Datasets e exemplos
├── requirements.txt                # Dependências
└── README.md                       # Documentação
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.9+
- CUDA (opcional, para GPU acceleration)
- Redis (para cache de dados)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/galafis/high-frequency-trading-analytics.git
cd high-frequency-trading-analytics
```

2. **Configure o ambiente:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Configure dados de mercado:**
```bash
# Configurar API keys para dados
export ALPHA_VANTAGE_API_KEY="your_api_key"
export POLYGON_API_KEY="your_api_key"
```

4. **Execute exemplo básico:**
```bash
python src/examples/basic_strategy.py
```

## 🤖 Reinforcement Learning para Trading

### Agente DQN (Deep Q-Network)
```python
from src.models.reinforcement_learning import DQNAgent

# Criar agente DQN
agent = DQNAgent(
    state_size=50,  # Features do mercado
    action_size=3,  # Buy, Sell, Hold
    learning_rate=0.001
)

# Treinar agente
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    agent.replay()  # Treinar rede neural
```

### Agente PPO (Proximal Policy Optimization)
```python
from src.models.reinforcement_learning import PPOAgent

# Agente PPO para trading contínuo
ppo_agent = PPOAgent(
    state_dim=50,
    action_dim=1,  # Posição contínua [-1, 1]
    lr_actor=0.0003,
    lr_critic=0.001
)

# Treinar com dados históricos
ppo_agent.train(
    market_data=historical_data,
    episodes=5000,
    max_steps=1000
)
```

## 📊 Estratégias de Trading

### Market Making
```python
from src.strategies.market_making import MarketMaker

# Estratégia de market making
mm_strategy = MarketMaker(
    spread_target=0.001,  # 10 bps
    inventory_limit=1000,
    risk_aversion=0.5
)

# Executar estratégia
orders = mm_strategy.generate_orders(
    current_price=100.50,
    order_book=order_book_data,
    inventory=current_inventory
)
```

### Arbitragem Estatística
```python
from src.strategies.arbitrage import StatisticalArbitrage

# Pairs trading
pairs_strategy = StatisticalArbitrage(
    lookback_window=252,  # 1 ano
    entry_threshold=2.0,  # 2 desvios padrão
    exit_threshold=0.5
)

# Identificar oportunidades
signals = pairs_strategy.generate_signals(
    asset1_prices=stock_a_prices,
    asset2_prices=stock_b_prices
)
```

## 📈 Análise de Microestrutura

### Order Book Analysis
```python
from src.data.order_book import OrderBookAnalyzer

# Analisar order book
ob_analyzer = OrderBookAnalyzer()

# Calcular métricas de liquidez
liquidity_metrics = ob_analyzer.calculate_liquidity(
    bids=order_book['bids'],
    asks=order_book['asks'],
    depth_levels=10
)

print(f"Bid-Ask Spread: {liquidity_metrics['spread']:.4f}")
print(f"Market Depth: {liquidity_metrics['depth']:.2f}")
print(f"Price Impact: {liquidity_metrics['impact']:.4f}")
```

### Market Impact Modeling
```python
from src.execution.market_impact import MarketImpactModel

# Modelo de impacto de mercado
impact_model = MarketImpactModel(
    model_type='linear',
    calibration_period=30  # dias
)

# Estimar impacto de ordem
estimated_impact = impact_model.estimate_impact(
    order_size=10000,
    average_daily_volume=1000000,
    volatility=0.02
)
```

## 🔍 Backtesting Framework

### Engine de Backtesting
```python
from src.backtesting.engine import BacktestEngine

# Configurar backtest
backtest = BacktestEngine(
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_capital=1000000,
    commission=0.001
)

# Adicionar estratégia
backtest.add_strategy(mm_strategy)

# Executar backtest
results = backtest.run()

# Analisar resultados
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Métricas de Performance
```python
from src.backtesting.metrics import PerformanceMetrics

# Calcular métricas avançadas
metrics = PerformanceMetrics(returns=strategy_returns)

performance_report = {
    'sharpe_ratio': metrics.sharpe_ratio(),
    'sortino_ratio': metrics.sortino_ratio(),
    'calmar_ratio': metrics.calmar_ratio(),
    'var_95': metrics.value_at_risk(confidence=0.95),
    'cvar_95': metrics.conditional_var(confidence=0.95)
}
```

## ⚡ Otimização de Performance

### Compilação JIT com Numba
```python
from numba import jit
import numpy as np

@jit(nopython=True)
def fast_moving_average(prices, window):
    """Moving average otimizada com Numba"""
    n = len(prices)
    ma = np.empty(n)
    
    for i in range(window-1, n):
        ma[i] = np.mean(prices[i-window+1:i+1])
    
    return ma

# Uso
fast_ma = fast_moving_average(price_data, 20)
```

### Processamento Paralelo
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_backtest(strategy_params):
    """Backtest paralelo para otimização de parâmetros"""
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []
        
        for params in strategy_params:
            future = executor.submit(run_single_backtest, params)
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return results
```

## 🔧 Configuração e Deploy

### Configuração de Produção
```python
# config/production.py
TRADING_CONFIG = {
    'max_position_size': 10000,
    'risk_limit': 0.02,  # 2% do capital
    'latency_threshold': 1000,  # microsegundos
    'data_frequency': '1ms'
}

EXECUTION_CONFIG = {
    'order_type': 'limit',
    'time_in_force': 'IOC',
    'max_slippage': 0.0005
}
```

### Monitoramento em Tempo Real
```python
from src.utils.monitoring import TradingMonitor

# Monitor de trading
monitor = TradingMonitor()

# Alertas automáticos
monitor.add_alert(
    metric='drawdown',
    threshold=0.05,  # 5%
    action='stop_trading'
)

monitor.add_alert(
    metric='latency',
    threshold=5000,  # 5ms
    action='switch_venue'
)
```

## 🧪 Testes e Validação

### Executar Testes
```bash
# Testes unitários
pytest tests/unit/

# Testes de integração
pytest tests/integration/

# Testes de performance
pytest tests/performance/

# Testes de estratégias
pytest tests/strategies/
```

### Validação de Modelos
```bash
# Validação cruzada de modelos
python scripts/validate_models.py --model dqn --periods 10

# Teste de robustez
python scripts/robustness_test.py --strategy market_making
```

## 📊 Casos de Uso Avançados

### 1. Crypto Market Making
- Market making em exchanges de criptomoedas
- Gestão de inventory multi-asset
- Arbitragem cross-exchange

### 2. Equity Statistical Arbitrage
- Pairs trading em ações
- Basket trading com ETFs
- Mean reversion strategies

### 3. FX High-Frequency Trading
- Trading em mercado de câmbio
- Carry trade automatizado
- News-based trading

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- Email: gabrieldemetrios@gmail.com

---

⭐ Se este projeto foi útil, considere deixar uma estrela!

