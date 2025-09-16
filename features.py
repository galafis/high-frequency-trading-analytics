"""
features.py - Pipeline de Feature Engineering para High Frequency Trading (HFT)
Crie, normalize, visualize e exporte features de mercado automaticamente.

Uso CLI:
python features.py --input ../data/raw/meusdados.csv --output_parquet ../data/processed/features.parquet --output_csv ../data/processed/features.csv
"""

import pandas as pd
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def feature_engineering(df):
    """Cria features clássicas de mercado para HFT."""
    df['price_lag1'] = df['price'].shift(1)
    df['log_return_1'] = np.log(df['price'] / df['price_lag1'])
    df['price_ma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
    df['vol_10'] = df['log_return_1'].rolling(window=10, min_periods=2).std()
    if {'ask', 'bid'}.issubset(df.columns):
        df['spread'] = df['ask'] - df['bid']
    if {'ask_volume', 'bid_volume'}.issubset(df.columns):
        df['obi'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-6)
    if {'order_qty', 'exec_qty'}.issubset(df.columns):
        df['partial_exec_flag'] = (df['exec_qty'] < df['order_qty']).astype(int)
    window = 300
    df['price_zscore_5min'] = (
        (df['price'] - df['price'].rolling(window, min_periods=30).mean()) /
        (df['price'].rolling(window, min_periods=30).std() + 1e-8)
    )
    df['log_return_scaled'] = (
        (df['log_return_1'] - df['log_return_1'].rolling(100, min_periods=20).min()) /
        (df['log_return_1'].rolling(100, min_periods=20).max() - df['log_return_1'].rolling(100, min_periods=20).min() + 1e-8)
    )
    return df

def export_features(df, out_parquet, out_csv):
    """Exporta features finais em Parquet e CSV."""
    feature_cols = [
        'price', 'price_lag1', 'log_return_1', 'price_ma_10', 'vol_10',
        'obi', 'spread', 'partial_exec_flag', 'price_zscore_5min', 'log_return_scaled'
    ]
    out = df[feature_cols].dropna()
    out.to_parquet(out_parquet)
    out.to_csv(out_csv)
    logging.info(f"Exportação concluída: {out.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline HFT")
    parser.add_argument("--input", required=True, help="CSV de entrada (deve conter 'timestamp' e 'price')")
    parser.add_argument("--output_parquet", required=True, help="Caminho do parquet de saída")
    parser.add_argument("--output_csv", required=True, help="Caminho do csv de saída")
    args = parser.parse_args()
    logging.info("Iniciando pipeline")
    df = pd.read_csv(args.input, parse_dates=['timestamp'], index_col='timestamp')
    df = feature_engineering(df)
    export_features(df, args.output_parquet, args.output_csv)
    logging.info("Pipeline finalizado com sucesso!")
