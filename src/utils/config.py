"""
config.py

Módulo de gerenciamento de configurações para o projeto High Frequency Trading Analytics.
Permite centralizar, carregar e validar parâmetros de ambiente por arquivo (YAML/JSON) ou variáveis de ambiente.

Suporta perfis (development, production, test), seed de experimentos, credenciais, caminhos de dados, hiperparâmetros etc.

Autor: Gabriel Demetrios Lafis
Data: setembro/2025
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml  # Para suporte YAML
except ImportError:
    yaml = None

class Config:
    """
    Centralizador de configurações do projeto.
    """

    def __init__(self, config_file: Optional[str] = None, env_prefix: str = "HFT_"):
        """
        Inicializa configuração a partir de arquivo e variáveis de ambiente.

        Args:
            config_file (str, opcional): Caminho absoluto para config.yaml ou config.json.
            env_prefix (str): Prefixo das variáveis de ambiente.
        """
        self.env_prefix = env_prefix
        self.data: Dict[str, Any] = {}

        if config_file:
            loaded = self._load_config_file(config_file)
            self.data.update(loaded)

        self._load_env_vars()


    def _load_config_file(self, config_file: str) -> dict:
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_file}")
        if config_file.endswith(".json"):
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        elif config_file.endswith((".yml", ".yaml")) and yaml:
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("Formato de arquivo não suportado (use YAML ou JSON)")

    def _load_env_vars(self):
        """
        Carrega variáveis de ambiente com prefixo definido.
        """
        for k, v in os.environ.items():
            if k.startswith(self.env_prefix):
                key = k[len(self.env_prefix):].lower()
                self.data[key] = v

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.data)


# Exemplo de uso
if __name__ == "__main__":
    # Criação/manual (ideal em testes)
    cfg = Config()
    cfg.set('initial_capital', 1_000_000)
    print("Config manual:", cfg.as_dict())

    # Carregando YAML, se pyyaml disponível:
    if yaml:
        print("Config YAML exemplo:")
        exemplo_yaml = """
        initial_capital: 2000000
        commission: 0.0007
        log_level: DEBUG
        """
        with open("config_example.yml", "w") as f:
            f.write(exemplo_yaml)
        cfg_yaml = Config("config_example.yml")
        print(cfg_yaml.as_dict())
