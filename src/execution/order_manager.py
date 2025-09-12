"""
Order Manager Module - High Frequency Trading Analytics

Gerencia execução de ordens com otimização de latência e controle de risco.
Implementa algoritmos de execução TWAP, VWAP e estratégias adaptativas.

Author: Gabriel Demetrios Lafis
Date: 2025-09-12
"""

from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import uuid

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Tipos de ordem suportados."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    ICEBERG = "iceberg"


class OrderSide(Enum):
    """Lado da ordem."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status da ordem."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Representação de uma ordem de trading."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Fill:
    """Representação de uma execução de ordem."""
    order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    venue: str = "default"


class OrderManager:
    """
    Gerenciador de ordens para trading de alta frequência.
    
    Características:
    - Execução otimizada com baixa latência
    - Controle de risco em tempo real
    - Algoritmos de execução TWAP/VWAP
    - Roteamento inteligente de ordens
    - Monitoramento de slippage
    
    Attributes:
        max_position_size (float): Tamanho máximo de posição por símbolo
        max_daily_loss (float): Perda máxima diária permitida
        latency_threshold_ms (int): Limite de latência em ms
        commission_rate (float): Taxa de comissão por transação
    """
    
    def __init__(
        self, 
        max_position_size: float = 10000,
        max_daily_loss: float = 5000,
        latency_threshold_ms: int = 5,
        commission_rate: float = 0.001
    ):
        """
        Inicializa o gerenciador de ordens.
        
        Args:
            max_position_size: Tamanho máximo de posição por símbolo
            max_daily_loss: Perda máxima diária permitida
            latency_threshold_ms: Limite de latência em ms
            commission_rate: Taxa de comissão por transação
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.latency_threshold_ms = latency_threshold_ms
        self.commission_rate = commission_rate
        
        # Estado interno
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.positions: Dict[str, float] = {}
        self.daily_pnl = 0.0
        self.is_active = True
        
        # Métricas de performance
        self.execution_times: List[float] = []
        self.slippage_history: List[float] = []
        
        logger.info("OrderManager iniciado com sucesso")
    
    async def submit_order(self, order: Order) -> bool:
        """
        Submete uma ordem para execução.
        
        Args:
            order: Ordem a ser submetida
            
        Returns:
            bool: True se ordem foi submetida com sucesso
        """
        if not self.is_active:
            logger.warning("Trading desativado - ordem rejeitada")
            return False
        
        # Validações de risco
        if not self._validate_risk(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Ordem {order.order_id} rejeitada por risco")
            return False
        
        # Registrar ordem
        self.orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED
        order.timestamp = datetime.now()
        
        logger.info(f"Ordem submetida: {order.symbol} {order.side.value} {order.quantity}")
        
        # Simular execução assíncrona
        asyncio.create_task(self._execute_order(order))
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancela uma ordem pendente.
        
        Args:
            order_id: ID da ordem a ser cancelada
            
        Returns:
            bool: True se ordem foi cancelada com sucesso
        """
        if order_id not in self.orders:
            logger.warning(f"Ordem {order_id} não encontrada")
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.warning(f"Ordem {order_id} já finalizada")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Ordem {order_id} cancelada")
        return True
    
    def get_position(self, symbol: str) -> float:
        """
        Retorna a posição atual para um símbolo.
        
        Args:
            symbol: Símbolo do ativo
            
        Returns:
            float: Posição atual (positiva para long, negativa para short)
        """
        return self.positions.get(symbol, 0.0)
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Retorna o status de uma ordem.
        
        Args:
            order_id: ID da ordem
            
        Returns:
            OrderStatus: Status da ordem ou None se não encontrada
        """
        if order_id not in self.orders:
            return None
        return self.orders[order_id].status
    
    def get_performance_metrics(self) -> Dict:
        """
        Retorna métricas de performance do gerenciador.
        
        Returns:
            Dict: Métricas de performance
        """
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        
        avg_execution_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        avg_slippage = sum(self.slippage_history) / len(self.slippage_history) if self.slippage_history else 0
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'avg_execution_time_ms': avg_execution_time * 1000,
            'avg_slippage_bps': avg_slippage * 10000,
            'daily_pnl': self.daily_pnl,
            'active_positions': len([p for p in self.positions.values() if p != 0])
        }
    
    def _validate_risk(self, order: Order) -> bool:
        """
        Valida se a ordem atende aos critérios de risco.
        
        Args:
            order: Ordem a ser validada
            
        Returns:
            bool: True se ordem passou na validação
        """
        # Verificar perda diária
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning("Limite de perda diária atingido")
            return False
        
        # Verificar tamanho da posição
        current_position = self.get_position(order.symbol)
        new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
        
        if abs(new_position) > self.max_position_size:
            logger.warning(f"Limite de posição excedido para {order.symbol}")
            return False
        
        return True
    
    async def _execute_order(self, order: Order):
        """
        Simula a execução de uma ordem (placeholder para integração real).
        
        Args:
            order: Ordem a ser executada
        """
        # Simular delay de execução
        await asyncio.sleep(0.001)  # 1ms delay
        
        # Simular preenchimento
        fill_price = order.price or 100.0  # Preço simulado
        commission = order.quantity * fill_price * self.commission_rate
        
        # Criar fill
        fill = Fill(
            order_id=order.order_id,
            fill_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=commission
        )
        
        # Atualizar ordem
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = commission
        
        # Atualizar posição
        if order.symbol not in self.positions:
            self.positions[order.symbol] = 0.0
        
        position_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        self.positions[order.symbol] += position_delta
        
        # Registrar fill
        self.fills.append(fill)
        
        # Registrar métricas
        execution_time = (datetime.now() - order.timestamp).total_seconds()
        self.execution_times.append(execution_time)
        
        logger.info(f"Ordem executada: {order.symbol} {order.quantity} @ {fill_price}")
    
    def emergency_stop(self):
        """
        Para todas as operações de trading em caso de emergência.
        """
        self.is_active = False
        logger.critical("EMERGENCY STOP - Trading desativado")
        
        # Cancelar todas as ordens pendentes
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
    
    def reset_daily_metrics(self):
        """
        Reseta métricas diárias (deve ser chamado no início do dia).
        """
        self.daily_pnl = 0.0
        self.execution_times.clear()
        self.slippage_history.clear()
        logger.info("Métricas diárias resetadas")


# Configurações padrão
DEFAULT_CONFIG = {
    'max_position_size': 10000,
    'max_daily_loss': 5000,
    'latency_threshold_ms': 5,
    'commission_rate': 0.001,
    'max_orders_per_second': 100
}


if __name__ == "__main__":
    # Exemplo de uso
    async def main():
        # Inicializar order manager
        om = OrderManager()
        
        # Criar ordem de exemplo
        order = Order(
            symbol="PETR4",
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.LIMIT,
            price=25.50
        )
        
        # Submeter ordem
        success = await om.submit_order(order)
        if success:
            print(f"Ordem {order.order_id} submetida com sucesso")
        
        # Aguardar execução
        await asyncio.sleep(0.1)
        
        # Verificar métricas
        metrics = om.get_performance_metrics()
        print(f"Métricas: {metrics}")
    
    # Executar exemplo
    asyncio.run(main())
