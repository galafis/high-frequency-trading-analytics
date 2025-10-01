# Data Directory

Este diretório contém datasets e exemplos de dados para o sistema de high-frequency trading analytics.

## Estrutura de Dados

### Dados de Mercado
- `market_data/` - Dados históricos de preços e volumes
- `order_book_data/` - Dados de order book em alta frequência
- `tick_data/` - Dados de tick por tick

### Dados de Exemplo
- `sample_data/` - Datasets de exemplo para testes
- `synthetic_data/` - Dados sintéticos para desenvolvimento

### Dados Processados
- `features/` - Features processadas para ML
- `cleaned_data/` - Dados limpos e tratados

## Formatos Suportados

- **CSV**: Dados tabulares básicos
- **Parquet**: Dados otimizados para análise
- **HDF5**: Datasets grandes com alta performance
- **JSON**: Dados de configuração e metadados

## Uso

```python
import pandas as pd
from src.data.market_data import MarketDataLoader

# Carregar dados de mercado
loader = MarketDataLoader()
data = loader.load_ohlcv('AAPL', '2023-01-01', '2023-12-31')

# Carregar dados de order book
ob_data = loader.load_order_book('AAPL', '2023-01-01')
```

## Fontes de Dados

- Alpha Vantage API
- Polygon.io API
- Yahoo Finance
- Quandl
- Dados proprietários de exchanges

## Estrutura de Arquivos

```
data/
├── market_data/
│   ├── equities/
│   ├── forex/
│   └── crypto/
├── order_book_data/
├── tick_data/
├── sample_data/
├── synthetic_data/
├── features/
└── cleaned_data/
```

## Configuração

Configure as APIs necessárias no arquivo `.env`:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
```

## Notas Importantes

- Dados sensíveis não devem ser commitados no repositório
- Use `.gitignore` para excluir arquivos de dados grandes
- Mantenha backups dos dados importantes
- Documente a origem e processamento de cada dataset
