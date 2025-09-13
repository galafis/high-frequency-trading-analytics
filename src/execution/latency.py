# -*- coding: utf-8 -*-
"""
Latency Optimization Module for High-Frequency Trading

This module provides tools for measuring, monitoring, and optimizing latency
in high-frequency trading systems. Critical for sub-millisecond execution.

Author: Gabriel Demetrios Lafis
Created: 2024
"""

import time
import statistics
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
from datetime import datetime, timedelta


@dataclass
class LatencyMeasurement:
    """Container for latency measurement data"""
    timestamp: datetime
    latency_us: float  # microseconds
    component: str
    metadata: Dict[str, Any]


class LatencyMonitor:
    """
    Real-time latency monitoring and analysis for HFT systems.
    
    Tracks latencies across different components and provides
    alerts when thresholds are exceeded.
    """
    
    def __init__(self, window_size: int = 1000, alert_threshold_us: float = 1000.0):
        self.window_size = window_size
        self.alert_threshold_us = alert_threshold_us
        self.measurements = deque(maxlen=window_size)
        self.component_stats = {}
        self.alert_callbacks = []
        self._lock = threading.Lock()
    
    def add_measurement(self, component: str, latency_us: float, metadata: Dict = None):
        """Add a new latency measurement"""
        measurement = LatencyMeasurement(
            timestamp=datetime.now(),
            latency_us=latency_us,
            component=component,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.measurements.append(measurement)
            self._update_component_stats(component, latency_us)
            
            # Check for alerts
            if latency_us > self.alert_threshold_us:
                self._trigger_alert(measurement)
    
    def _update_component_stats(self, component: str, latency_us: float):
        """Update statistics for a specific component"""
        if component not in self.component_stats:
            self.component_stats[component] = deque(maxlen=self.window_size)
        
        self.component_stats[component].append(latency_us)
    
    def get_statistics(self, component: Optional[str] = None) -> Dict:
        """Get latency statistics for all components or a specific one"""
        with self._lock:
            if component:
                if component in self.component_stats:
                    data = list(self.component_stats[component])
                    return self._calculate_stats(data)
                return {}
            
            stats = {}
            for comp, data in self.component_stats.items():
                stats[comp] = self._calculate_stats(list(data))
            
            return stats
    
    def _calculate_stats(self, data: List[float]) -> Dict:
        """Calculate statistical measures for latency data"""
        if not data:
            return {}
        
        return {
            'count': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'min': min(data),
            'max': max(data),
            'p95': np.percentile(data, 95),
            'p99': np.percentile(data, 99),
            'std': statistics.stdev(data) if len(data) > 1 else 0.0
        }
    
    def add_alert_callback(self, callback: Callable[[LatencyMeasurement], None]):
        """Add callback function for latency alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, measurement: LatencyMeasurement):
        """Trigger alert callbacks when threshold is exceeded"""
        for callback in self.alert_callbacks:
            try:
                callback(measurement)
            except Exception as e:
                print(f"Error in alert callback: {e}")


class LatencyProfiler:
    """
    Context manager for profiling function execution latency.
    
    Usage:
        with LatencyProfiler('order_processing') as profiler:
            # Your code here
            pass
    """
    
    def __init__(self, component_name: str, monitor: Optional[LatencyMonitor] = None):
        self.component_name = component_name
        self.monitor = monitor
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        latency_us = (self.end_time - self.start_time) * 1_000_000
        
        if self.monitor:
            self.monitor.add_measurement(self.component_name, latency_us)
    
    @property
    def latency_microseconds(self) -> Optional[float]:
        """Get the measured latency in microseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1_000_000
        return None


def latency_decorator(component_name: str, monitor: Optional[LatencyMonitor] = None):
    """
    Decorator for measuring function execution latency.
    
    Args:
        component_name: Name to identify the component
        monitor: LatencyMonitor instance to record measurements
    
    Usage:
        @latency_decorator('market_data_processing')
        def process_market_data(data):
            # Your processing logic
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LatencyProfiler(component_name, monitor):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class LatencyOptimizer:
    """
    Provides optimization techniques for reducing system latency.
    """
    
    @staticmethod
    def cpu_affinity_optimization():
        """
        Set CPU affinity for critical trading threads.
        Pins threads to specific CPU cores to reduce context switching.
        """
        import os
        try:
            # Pin to the first CPU core (adjust as needed)
            os.sched_setaffinity(0, {0})
            print("CPU affinity set to core 0")
        except AttributeError:
            print("CPU affinity not supported on this platform")
    
    @staticmethod
    def memory_optimization():
        """
        Pre-allocate memory pools to avoid garbage collection delays.
        """
        # Example: Pre-allocate numpy arrays
        price_buffer = np.zeros(10000, dtype=np.float64)
        volume_buffer = np.zeros(10000, dtype=np.int64)
        
        return {
            'price_buffer': price_buffer,
            'volume_buffer': volume_buffer
        }
    
    @staticmethod
    def network_optimization_tips() -> List[str]:
        """
        Returns a list of network optimization recommendations.
        """
        return [
            "Use kernel bypass networking (DPDK, user-space TCP)",
            "Implement UDP multicast for market data",
            "Co-locate servers near exchange data centers",
            "Use dedicated network interfaces for trading traffic",
            "Implement custom network protocols for minimal overhead",
            "Use RDMA (InfiniBand) for ultra-low latency",
            "Optimize network buffers and interrupt handling"
        ]


class TimestampManager:
    """
    High-precision timestamp management for latency measurements.
    """
    
    @staticmethod
    def get_hardware_timestamp() -> int:
        """
        Get hardware timestamp using CPU cycle counter.
        Most precise timing available.
        """
        import time
        return time.perf_counter_ns()
    
    @staticmethod
    def get_monotonic_timestamp() -> float:
        """
        Get monotonic timestamp in microseconds.
        Suitable for latency measurements.
        """
        return time.monotonic() * 1_000_000
    
    @staticmethod
    def convert_ns_to_us(timestamp_ns: int) -> float:
        """Convert nanoseconds to microseconds"""
        return timestamp_ns / 1000.0


class LatencyBenchmark:
    """
    Benchmark different operations to establish latency baselines.
    """
    
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations
    
    def benchmark_function_call(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Benchmark a function call latency.
        """
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            latencies.append((end - start) * 1_000_000)  # Convert to microseconds
        
        return {
            'mean_latency_us': statistics.mean(latencies),
            'median_latency_us': statistics.median(latencies),
            'min_latency_us': min(latencies),
            'max_latency_us': max(latencies),
            'p95_latency_us': np.percentile(latencies, 95),
            'p99_latency_us': np.percentile(latencies, 99)
        }
    
    def benchmark_memory_allocation(self, size: int) -> Dict:
        """
        Benchmark memory allocation latency.
        """
        def allocate_memory():
            data = [0] * size
            return data
        
        return self.benchmark_function_call(allocate_memory)
    
    def benchmark_arithmetic_operations(self) -> Dict:
        """
        Benchmark basic arithmetic operations.
        """
        def arithmetic_ops():
            a = 100.5
            b = 200.7
            result = (a + b) * (a - b) / (a + 1)
            return result
        
        return self.benchmark_function_call(arithmetic_ops)


# Example usage and testing
if __name__ == "__main__":
    # Create latency monitor
    monitor = LatencyMonitor(window_size=1000, alert_threshold_us=500.0)
    
    # Add alert callback
    def alert_handler(measurement: LatencyMeasurement):
        print(f"ALERT: High latency detected in {measurement.component}: "
              f"{measurement.latency_us:.2f}μs")
    
    monitor.add_alert_callback(alert_handler)
    
    # Example profiling
    @latency_decorator('market_data_processing', monitor)
    def process_market_data():
        time.sleep(0.0001)  # Simulate processing
    
    # Run some tests
    for i in range(10):
        process_market_data()
        
        # Simulate varying latencies
        import random
        latency = random.uniform(100, 1000)
        monitor.add_measurement('order_execution', latency)
    
    # Print statistics
    stats = monitor.get_statistics()
    for component, stat in stats.items():
        print(f"\n{component} latency statistics:")
        for metric, value in stat.items():
            print(f"  {metric}: {value:.2f}μs")
    
    # Run benchmarks
    benchmark = LatencyBenchmark(iterations=1000)
    
    print("\nBenchmark Results:")
    print("Memory allocation:", benchmark.benchmark_memory_allocation(1000))
    print("Arithmetic operations:", benchmark.benchmark_arithmetic_operations())
