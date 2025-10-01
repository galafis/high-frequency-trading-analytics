
import pytest
import pandas as pd
import os
from pathlib import Path
import sys

# Adicionar o diretório raiz ao path para importar validate_data.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validate_data import validate_csv_schema


class TestValidateData:
    """Classe de testes para validar o script validate_data.py."""

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Fixture para criar um arquivo CSV temporário."""
        def _create_csv(content, filename="test_data.csv"):
            file_path = tmp_path / filename
            file_path.write_text(content)
            return str(file_path)
        return _create_csv

    def test_file_not_found(self):
        """Testa se FileNotFoundError é levantado para arquivo inexistente."""
        with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
            validate_csv_schema("non_existent_file.csv")

    def test_empty_csv(self, temp_csv_file):
        """Testa se AssertionError é levantado para arquivo CSV vazio."""
        file_path = temp_csv_file("")
        with pytest.raises(AssertionError, match="Arquivo CSV está vazio"):
            validate_csv_schema(file_path)

    def test_missing_required_columns(self, temp_csv_file):
        """Testa se AssertionError é levantado para colunas obrigatórias ausentes."""
        content = "col1,col2\n1,2"
        file_path = temp_csv_file(content)
        with pytest.raises(AssertionError, match="Colunas obrigatórias ausentes: \['timestamp', 'price'\]"):
            validate_csv_schema(file_path)

    def test_null_values_in_required_columns(self, temp_csv_file):
        """Testa se AssertionError é levantado para valores nulos em colunas obrigatórias."""
        content = "timestamp,price\n2025-01-01 09:00:00,100.0\n,101.0\n2025-01-01 09:00:02,"
        file_path = temp_csv_file(content)
        with pytest.raises(AssertionError, match="Valores nulos encontrados nas colunas"):
            validate_csv_schema(file_path)

    def test_invalid_timestamp_format(self, temp_csv_file):
        """Testa se AssertionError é levantado para formato de timestamp inválido."""
        content = "timestamp,price\nnot-a-date,100.0"
        file_path = temp_csv_file(content)
        with pytest.raises(AssertionError, match="Coluna 'timestamp' não está em formato de data válido"):
            validate_csv_schema(file_path)

    def test_non_numeric_price(self, temp_csv_file):
        """Testa se AssertionError é levantado para preço não numérico."""
        content = "timestamp,price\n2025-01-01 09:00:00,abc"
        file_path = temp_csv_file(content)
        with pytest.raises(AssertionError, match="Coluna 'price' deve conter valores numéricos"):
            validate_csv_schema(file_path)

    def test_negative_price(self, temp_csv_file):
        """Testa se AssertionError é levantado para preço negativo."""
        content = "timestamp,price\n2025-01-01 09:00:00,-10.0"
        file_path = temp_csv_file(content)
        with pytest.raises(AssertionError, match="Valores de preço não podem ser negativos"):
            validate_csv_schema(file_path)

    def test_valid_csv(self, temp_csv_file):
        """Testa um arquivo CSV válido."""
        content = "timestamp,price\n2025-01-01 09:00:00,100.0\n2025-01-01 09:00:01,101.0"
        file_path = temp_csv_file(content)
        assert validate_csv_schema(file_path) is True

    def test_valid_csv_with_extra_columns(self, temp_csv_file):
        """Testa um arquivo CSV válido com colunas extras."""
        content = "timestamp,price,volume\n2025-01-01 09:00:00,100.0,1000\n2025-01-01 09:00:01,101.0,1200"
        file_path = temp_csv_file(content)
        assert validate_csv_schema(file_path) is True


