"""
logger.py

Logger customizável do projeto, suporta logging em console, arquivo e diferentes níveis para cada subsistema (dados, execução, risco, estratégias, etc).
Pensado para integração com sistemas produtivos, CI/CD e auditoria.

Autor: Gabriel Demetrios Lafis
Data: setembro/2025
"""

import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "hft_analytics",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configura um logger customizado para o projeto.

    Args:
        name (str): Nome identificador do logger.
        level (int): Nível mínimo de logging.
        log_file (str): Caminho para arquivo de log.
        console (bool): Se True, ativa output para console (stdout).

    Returns:
        logging.Logger: Logger pronto para uso.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Remover handlers duplicados em caso de re-execução em notebooks
    if logger.hasHandlers():
        logger.handlers.clear()

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

# Exemplo de uso
if __name__ == "__main__":
    logger = setup_logger("exemplo", log_file="hft_analytics.log")
    logger.info("Logger de exemplo funcionando!")
    logger.warning("Aviso de exemplo.")
    logger.error("Erro de exemplo.")
