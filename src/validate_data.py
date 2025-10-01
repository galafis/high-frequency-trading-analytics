#!/usr/bin/env python3
"""
Script de Validação de Dados CSV para High-Frequency Trading Analytics

Este script valida se arquivos CSV possuem o schema necessário para análise de dados
de trading de alta frequência. Verifica se contém as colunas ['timestamp', 'price']
e garante a ausência de valores nulos nessas colunas críticas.

Uso:
    python validate_data.py <arquivo.csv>

Exemplo:
    python validate_data.py data/market_data.csv

Autor: Gabriel Demetrios Lafis
Data: Setembro 2025
"""

import sys
import pandas as pd
from pathlib import Path


def validate_csv_schema(file_path: str) -> bool:
    """
    Valida o schema de um arquivo CSV de dados de mercado.
    
    Args:
        file_path (str): Caminho para o arquivo CSV a ser validado
        
    Returns:
        bool: True se o arquivo é válido, False caso contrário
        
    Raises:
        AssertionError: Se o arquivo não atender aos requisitos de schema
        FileNotFoundError: Se o arquivo não existir
        Exception: Para outros erros durante a leitura
    """
    
    # Verificar se o arquivo existe
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    try:
        # Ler o arquivo CSV
        df = pd.read_csv(file_path)
        
        # Verificar se o DataFrame não está vazio
        assert not df.empty, "Arquivo CSV está vazio"
        
        # Verificar se contém as colunas necessárias
        required_columns = ['timestamp', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        assert not missing_columns, f"Colunas obrigatórias ausentes: {missing_columns}"
        
        # Verificar se há valores nulos nas colunas críticas
        null_counts = df[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0].index.tolist()
        
        assert not columns_with_nulls, f"Valores nulos encontrados nas colunas: {columns_with_nulls}"
        
        # Validações adicionais opcionais
        # Verificar se timestamp está em formato adequado
        try:
            pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise AssertionError(f"Coluna 'timestamp' não está em formato de data válido: {str(e)}")
        
        # Verificar se price é numérico
        if not pd.api.types.is_numeric_dtype(df['price']):
            raise AssertionError("Coluna 'price' deve conter valores numéricos")
        
        # Verificar se há preços negativos (opcional, pode ser removido se necessário)
        if (df['price'] < 0).any():
            raise AssertionError("Valores de preço não podem ser negativos")
        
        return True
        
    except pd.errors.EmptyDataError:
        raise AssertionError("Arquivo CSV está vazio ou corrompido")
    except pd.errors.ParserError as e:
        raise AssertionError(f"Erro ao analisar CSV: {str(e)}")
    except Exception as e:
        raise AssertionError(f"Erro inesperado durante validação: {str(e)}")


def main():
    """
    Função principal que processa argumentos da linha de comando e executa validação.
    """
    
    # Verificar argumentos da linha de comando
    if len(sys.argv) != 2:
        print("Uso: python validate_data.py <arquivo.csv>")
        print("Exemplo: python validate_data.py data/market_data.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Executar validação
        validate_csv_schema(file_path)
        
        # Se chegou aqui, validação passou
        print("Validação OK!")
        sys.exit(0)
        
    except AssertionError as e:
        print(f"Erro de validação: {str(e)}")
        raise
        
    except FileNotFoundError as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
