"""
High-Frequency Trading Analytics - Logging Utilities

This module provides a comprehensive logging setup for the HFT Analytics project,
supporting console, file, and modular logging configurations.

Compatible with Python 3.9+

Author: HFT Analytics Team
Date: September 2025
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color coding to log messages for console output.
    
    This formatter enhances readability by applying different colors to different
    log levels when outputting to a terminal that supports ANSI color codes.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with color coding.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message with color codes
        """
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    log_level: Union[str, int] = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up a comprehensive logger for the HFT Analytics project.
    
    This function creates a logger with customizable console and file output,
    automatic log rotation, and modular configuration suitable for different
    components of the HFT Analytics system.
    
    Args:
        name: Name of the logger (typically __name__ of the module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, uses 'logs' in project root
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup log files to keep
        log_format: Custom log format string
        date_format: Custom date format string
        use_colors: Whether to use colored output in console
        
    Returns:
        Configured logger instance
        
    Raises:
        OSError: If log directory cannot be created
        ValueError: If invalid log level is provided
        
    Example:
        >>> logger = setup_logger(__name__, log_level='DEBUG')
        >>> logger.info("Starting HFT Analytics module")
        >>> logger.error("Failed to process market data", exc_info=True)
    """
    
    # Validate log level
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), None)
        if log_level is None:
            raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Set log level
    logger.setLevel(log_level)
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    # Default date format
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format, date_format)
        else:
            console_formatter = logging.Formatter(log_format, date_format)
            
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        # Set up log directory
        if log_dir is None:
            # Default to 'logs' directory in project root
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create log directory {log_dir}: {e}")
        
        # Create log filename with module name and timestamp
        log_filename = f"{name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = log_dir / log_filename
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_filepath,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # File formatter (no colors)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to parent loggers
    logger.propagate = False
    
    return logger


def get_module_logger(
    module_name: str,
    config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get a logger configured specifically for a module with optional custom config.
    
    Args:
        module_name: Name of the module (typically __name__)
        config: Optional dictionary with custom configuration parameters
        
    Returns:
        Configured logger instance for the module
        
    Example:
        >>> # Basic usage
        >>> logger = get_module_logger(__name__)
        >>> 
        >>> # With custom configuration
        >>> config = {'log_level': 'DEBUG', 'use_colors': False}
        >>> logger = get_module_logger(__name__, config)
    """
    if config is None:
        config = {}
    
    return setup_logger(module_name, **config)


def setup_trading_logger(
    strategy_name: str,
    log_level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Set up a specialized logger for trading strategies.
    
    This creates a logger with trading-specific formatting and separate
    log files for better organization of trading-related logs.
    
    Args:
        strategy_name: Name of the trading strategy
        log_level: Logging level for the strategy
        
    Returns:
        Configured trading logger
        
    Example:
        >>> logger = setup_trading_logger("momentum_strategy")
        >>> logger.info("Strategy initialized with parameters: %s", params)
        >>> logger.warning("Risk limit approached: %.2f%%", risk_percentage)
    """
    trading_format = (
        "%(asctime)s - TRADING - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Create trading-specific log directory
    project_root = Path(__file__).parent.parent.parent
    trading_log_dir = project_root / "logs" / "trading"
    
    return setup_logger(
        f"trading.{strategy_name}",
        log_level=log_level,
        log_dir=trading_log_dir,
        log_format=trading_format
    )


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the logging utilities.
    
    This section demonstrates various ways to use the logging setup
    for different components of the HFT Analytics system.
    """
    
    print("=== HFT Analytics Logger Demo ===")
    
    # Basic logger setup
    print("\n1. Basic Logger Setup:")
    basic_logger = setup_logger("hft_analytics.demo")
    basic_logger.info("HFT Analytics system starting up")
    basic_logger.debug("This debug message might not show depending on log level")
    basic_logger.warning("Market volatility detected")
    basic_logger.error("Failed to connect to data feed")
    
    # Custom configuration
    print("\n2. Custom Configuration:")
    custom_logger = setup_logger(
        "hft_analytics.custom",
        log_level="DEBUG",
        use_colors=True,
        console_output=True,
        file_output=True
    )
    custom_logger.debug("Debug mode enabled")
    custom_logger.info("Custom logger initialized")
    
    # Module-specific logger
    print("\n3. Module-Specific Logger:")
    module_config = {
        "log_level": "INFO",
        "use_colors": True,
        "console_output": True
    }
    module_logger = get_module_logger("hft_analytics.data_processor", module_config)
    module_logger.info("Processing market data batch")
    module_logger.warning("Data quality issue detected in batch")
    
    # Trading strategy logger
    print("\n4. Trading Strategy Logger:")
    strategy_logger = setup_trading_logger("momentum_v1", "DEBUG")
    strategy_logger.info("Momentum strategy initialized")
    strategy_logger.debug("Signal strength: 0.85")
    strategy_logger.warning("Position size approaching limit")
    strategy_logger.error("Order execution failed")
    
    # Demonstrate exception logging
    print("\n5. Exception Logging:")
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError:
        basic_logger.error(
            "Mathematical error occurred",
            exc_info=True  # This includes the full traceback
        )
    
    # Performance logging example
    print("\n6. Performance Logging:")
    import time
    
    perf_logger = setup_logger("hft_analytics.performance")
    start_time = time.time()
    
    # Simulate some processing
    time.sleep(0.1)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    perf_logger.info(
        "Market data processing completed in %.3f seconds",
        processing_time
    )
    
    print("\n=== Demo Complete ===")
    print("Check the 'logs' directory for generated log files.")
