# 📊 High Frequency Trading Analytics

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D.svg)](https://redis.io/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**High Frequency Trading Analytics** — Real-time analytics platform for high-frequency trading data. Processes tick-level data with ultra-low latency for market microstructure insights and trading performance analysis.

Total source lines: **4,808** across **36** files in **4** languages.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+


#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/high-frequency-trading-analytics.git
cd high-frequency-trading-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```





### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
high-frequency-trading-analytics/
├── config/
│   ├── config.py
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── images/
│   │   └── README.md
│   ├── notebooks/
│   │   └── README.md
│   ├── architecture_diagram.md
│   ├── data_README.md
│   └── tests_README.md
├── logs/
│   └── README.md
├── src/
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── market_data.py
│   │   └── order_book.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── latency.py
│   │   ├── order_manager.py
│   │   └── slippage.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictive_models.py
│   │   ├── reinforcement_learning.py
│   │   └── risk_models.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── validate_models.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── arbitrage.py
│   │   ├── market_making.py
│   │   └── momentum.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── logger.py
│   ├── __init__.py
│   ├── dashboard.py
│   └── validate_data.py
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   └── test_validate_data.py
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
└── script.js
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 33 files |
| HTML | 1 files |
| JavaScript | 1 files |
| CSS | 1 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**High Frequency Trading Analytics** — Real-time analytics platform for high-frequency trading data. Processes tick-level data with ultra-low latency for market microstructure insights and trading performance analysis.

Total de linhas de código: **4,808** em **36** arquivos em **4** linguagens.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+


#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/high-frequency-trading-analytics.git
cd high-frequency-trading-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```




### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
high-frequency-trading-analytics/
├── config/
│   ├── config.py
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── images/
│   │   └── README.md
│   ├── notebooks/
│   │   └── README.md
│   ├── architecture_diagram.md
│   ├── data_README.md
│   └── tests_README.md
├── logs/
│   └── README.md
├── src/
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── market_data.py
│   │   └── order_book.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── latency.py
│   │   ├── order_manager.py
│   │   └── slippage.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictive_models.py
│   │   ├── reinforcement_learning.py
│   │   └── risk_models.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── validate_models.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── arbitrage.py
│   │   ├── market_making.py
│   │   └── momentum.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── logger.py
│   ├── __init__.py
│   ├── dashboard.py
│   └── validate_data.py
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   └── test_validate_data.py
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
└── script.js
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 33 files |
| HTML | 1 files |
| JavaScript | 1 files |
| CSS | 1 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
