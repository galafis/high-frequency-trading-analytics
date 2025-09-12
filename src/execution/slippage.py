"""
slippage.py

Módulo responsável pela modelagem e cálculo da slippage em execuções de ordens.
Slippage representa a diferença entre o preço esperado de uma transação e o preço efetivamente executado devido à liquidez de mercado, atrasos e impacto.

Funções principais:
- calcular_slippage: Cálculo de slippage baseado no volume da ordem e liquidez.
- simular_execucao: Simulação de execução mostrando efeito prático da slippage.

Autor: Gabriel Demetrios Lafis
Data: setembro/2025
"""

from typing import Optional

def calcular_slippage(preco_teorico: float, volume: float, liquidez: float, fator_slippage: float = 0.0005) -> float:
    """
    Calcula o valor absoluto de slippage para uma ordem.

    Args:
        preco_teorico (float): Preço esperado da ordem sem impacto.
        volume (float): Volume da ordem (unidades).
        liquidez (float): Liquidez disponível próxima ao book (unidades).
        fator_slippage (float): Fator ajustável do modelo de slippage.

    Returns:
        float: Valor da slippage a ser acrescido/subtraído do preço.
    """
