"""
test_features.py - Testes Automáticos para o Pipeline de Feature Engineering

Este arquivo contém testes automatizados pytest para validar o funcionamento
do pipeline de feature engineering (features.py) usado no sistema de
high-frequency trading analytics.

Executar testes:
    pytest tests/test_features.py -v
    pytest tests/test_features.py::TestFeatureEngineering::test_basic_features -v
    pytest tests/test_features.py --cov=features --cov-report=html

Pré-requisitos:
    pip install pytest pandas numpy tempfile
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Adicionar o diretório raiz ao path para importar features.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.features import feature_engineering, export_features


class TestFeatureEngineering:
    """Classe de testes para validar o pipeline de feature engineering."""
    
    def create_sample_data(self, num_rows=100):
        """Cria dados de amostra para testes com preços simulados."""
        np.random.seed(42)  # Para reprodutibilidade
        
        # Gerar timestamps
        start_time = datetime(2024, 1, 1, 9, 30, 0)
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_rows)]
        
        # Gerar preços com random walk
        initial_price = 100.0
        returns = np.random.normal(0, 0.001, num_rows)  # Retornos com volatilidade baixa
        log_returns = returns
        prices = [initial_price]
        
        for i in range(1, num_rows):
            prices.append(prices[-1] * np.exp(log_returns[i]))
        
        # Criar DataFrame básico
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        
        # Adicionar dados de bid/ask opcionais
        df['bid'] = df['price'] - 0.01
        df['ask'] = df['price'] + 0.01
        df['bid_volume'] = np.random.randint(100, 1000, num_rows)
        df['ask_volume'] = np.random.randint(100, 1000, num_rows)
        
        # Adicionar dados de ordem opcionais
        df['order_qty'] = np.random.randint(10, 100, num_rows)
        df['exec_qty'] = df['order_qty'] * np.random.uniform(0.5, 1.0, num_rows)
        df['exec_qty'] = df['exec_qty'].astype(int)
        
        df = df.set_index('timestamp')
        return df
    
    def test_basic_features_creation(self):
        """Testa se as features básicas são criadas corretamente."""
        df = self.create_sample_data(50)
        
        # Aplicar feature engineering
        df_features = feature_engineering(df)
        
        # Verificar se features principais foram criadas
        expected_features = ['price_lag1', 'log_return_1', 'price_ma_10', 'vol_10']
        
        for feature in expected_features:
            assert feature in df_features.columns, f"Feature '{feature}' não foi criada"
            assert not df_features[feature].isna().all(), f"Feature '{feature}' contém apenas NaN"
    
    def test_price_lag1_correctness(self):
        """Testa se price_lag1 é o preço com lag de 1 período."""
        df = self.create_sample_data(20)
        df_features = feature_engineering(df)
        
        # Verificar se price_lag1 é realmente o preço deslocado
        assert pd.isna(df_features['price_lag1'].iloc[0]), "Primeiro valor de price_lag1 deveria ser NaN"
        
        for i in range(1, len(df_features)):
            expected_lag1 = df_features['price'].iloc[i-1]
            actual_lag1 = df_features['price_lag1'].iloc[i]
            assert np.isclose(expected_lag1, actual_lag1), f"price_lag1 incorreto na linha {i}"
    
    def test_log_return_1_validity(self):
        """Testa se log_return_1 é calculado corretamente e não contém NaN (exceto primeira linha)."""
        df = self.create_sample_data(30)
        df_features = feature_engineering(df)
        
        # Primeira linha deve ser NaN (não há preço anterior)
        assert pd.isna(df_features['log_return_1'].iloc[0]), "Primeiro valor de log_return_1 deveria ser NaN"
        
        # Restante das linhas não deve conter NaN
        non_first_returns = df_features['log_return_1'].iloc[1:]
        assert not non_first_returns.isna().any(), "log_return_1 contém NaN após primeira linha"
        
        # Verificar se o cálculo está correto
        for i in range(1, min(5, len(df_features))):  # Testar primeiras 5 linhas
            expected_return = np.log(df_features['price'].iloc[i] / df_features['price_lag1'].iloc[i])
            actual_return = df_features['log_return_1'].iloc[i]
            assert np.isclose(expected_return, actual_return, rtol=1e-10), f"log_return_1 incorreto na linha {i}"
    
    def test_moving_average_calculation(self):
        """Testa se a média móvel de 10 períodos está correta."""
        df = self.create_sample_data(25)
        df_features = feature_engineering(df)
        
        # Para as primeiras 10 observações, verificar se usa min_periods=1
        for i in range(10):
            expected_ma = df_features['price'].iloc[:i+1].mean()
            actual_ma = df_features['price_ma_10'].iloc[i]
            assert np.isclose(expected_ma, actual_ma), f"Média móvel incorreta na linha {i}"
        
        # Para observações posteriores, verificar janela de 10 períodos
        for i in range(10, min(15, len(df_features))):
            expected_ma = df_features['price'].iloc[i-9:i+1].mean()
            actual_ma = df_features['price_ma_10'].iloc[i]
            assert np.isclose(expected_ma, actual_ma), f"Média móvel incorreta na linha {i}"
    
    def test_volatility_calculation(self):
        """Testa se a volatilidade vol_10 é calculada corretamente."""
        df = self.create_sample_data(25)
        df_features = feature_engineering(df)
        
        # Primeira linha deve ser NaN (não há retornos anteriores suficientes)
        assert pd.isna(df_features['vol_10'].iloc[0]), "Primeiro valor de vol_10 deveria ser NaN"
        
        # Segundo valor também deve ser NaN (min_periods=2)
        assert pd.isna(df_features['vol_10'].iloc[1]), "Segundo valor de vol_10 deveria ser NaN"
        
        # Verificar se não há NaN após período inicial
        non_initial_vol = df_features['vol_10'].iloc[2:]
        assert not non_initial_vol.isna().any(), "vol_10 contém NaN após período inicial"
    
    def test_optional_features_with_bid_ask(self):
        """Testa se features opcionais (spread, obi) são criadas quando bid/ask existem."""
        df = self.create_sample_data(20)
        df_features = feature_engineering(df)
        
        # Verificar se spread foi criado
        assert 'spread' in df_features.columns, "Feature 'spread' não foi criada"
        
        # Verificar se spread é calculado corretamente
        for i in range(5):
            expected_spread = df_features['ask'].iloc[i] - df_features['bid'].iloc[i]
            actual_spread = df_features['spread'].iloc[i]
            assert np.isclose(expected_spread, actual_spread), f"Spread incorreto na linha {i}"
        
        # Verificar se OBI foi criado
        assert 'obi' in df_features.columns, "Feature 'obi' não foi criada"
        assert not df_features['obi'].isna().any(), "OBI contém valores NaN"
    
    def test_optional_features_with_order_data(self):
        """Testa se features opcionais (partial_exec_flag) são criadas quando dados de ordem existem."""
        df = self.create_sample_data(20)
        df_features = feature_engineering(df)
        
        # Verificar se partial_exec_flag foi criado
        assert 'partial_exec_flag' in df_features.columns, "Feature 'partial_exec_flag' não foi criada"
        
        # Verificar se valores são 0 ou 1
        unique_values = set(df_features['partial_exec_flag'].unique())
        assert unique_values.issubset({0, 1}), "partial_exec_flag deve conter apenas 0 ou 1"
    
    def test_zscore_and_scaling_features(self):
        """Testa se features de zscore e scaling são criadas."""
        df = self.create_sample_data(350)  # Dados suficientes para janela de 300
        df_features = feature_engineering(df)
        
        # Verificar se features foram criadas
        assert 'price_zscore_5min' in df_features.columns, "Feature 'price_zscore_5min' não foi criada"
        assert 'log_return_scaled' in df_features.columns, "Feature 'log_return_scaled' não foi criada"
        
        # Verificar se não há muitos NaN no final da série
        non_nan_zscore = df_features['price_zscore_5min'].dropna()
        assert len(non_nan_zscore) > 50, "Muitos valores NaN em price_zscore_5min"
    
    def test_export_features_functionality(self):
        """Testa se a função export_features executa sem erro e cria arquivos."""
        df = self.create_sample_data(50)
        df_features = feature_engineering(df)
        
        # Criar arquivos temporários
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = os.path.join(temp_dir, 'test_features.parquet')
            csv_path = os.path.join(temp_dir, 'test_features.csv')
            
            # Executar export (não deve gerar erro)
            try:
                export_features(df_features, parquet_path, csv_path)
                export_success = True
            except Exception as e:
                export_success = False
                print(f"Erro na exportação: {e}")
            
            assert export_success, "Função export_features falhou"
            
            # Verificar se arquivos foram criados
            assert os.path.exists(parquet_path), "Arquivo parquet não foi criado"
            assert os.path.exists(csv_path), "Arquivo CSV não foi criado"
            
            # Verificar se arquivos não estão vazios
            assert os.path.getsize(parquet_path) > 0, "Arquivo parquet está vazio"
            assert os.path.getsize(csv_path) > 0, "Arquivo CSV está vazio"
    
    def test_export_features_content_validation(self):
        """Testa se os dados exportados são consistentes e sem NaN críticos."""
        df = self.create_sample_data(100)
        df_features = feature_engineering(df)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = os.path.join(temp_dir, 'test_features.parquet')
            csv_path = os.path.join(temp_dir, 'test_features.csv')
            
            export_features(df_features, parquet_path, csv_path)
            
            # Ler arquivos exportados
            df_from_parquet = pd.read_parquet(parquet_path)
            df_from_csv = pd.read_csv(csv_path, index_col=0)
            
            # Verificar se dados são consistentes
            assert len(df_from_parquet) > 0, "Dados exportados estão vazios"
            assert len(df_from_csv) > 0, "Dados exportados estão vazios"
            
            # Verificar se não há NaN nos dados finais
            assert not df_from_parquet.isna().any().any(), "Dados exportados contêm NaN"
            assert not df_from_csv.isna().any().any(), "Dados exportados contêm NaN"
    
    def test_minimal_data_requirements(self):
        """Testa comportamento com dados mínimos."""
        # Criar dataset muito pequeno
        df = self.create_sample_data(5)
        
        # Não deve gerar erro mesmo com poucos dados
        try:
            df_features = feature_engineering(df)
            minimal_data_success = True
        except Exception as e:
            minimal_data_success = False
            print(f"Erro com dados mínimos: {e}")
        
        assert minimal_data_success, "Pipeline falhou com dados mínimos"
        
        # Verificar se features principais ainda existem
        assert 'price_lag1' in df_features.columns
        assert 'log_return_1' in df_features.columns


if __name__ == '__main__':
    # Permitir execução direta do arquivo de teste
    pytest.main([__file__, '-v'])
