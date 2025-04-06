# File: config.py

import os
from datetime import datetime, time

# MT5 Connection Settings
MT5_TIMEZONE = "Europe/Moscow"  # MT5 server is on Moscow time (UTC+3)
MT5_TO_NY_OFFSET = 6  # Direct hour difference between MT5 and NY during EDT

# Define timeframes
TIMEFRAMES = {
    "MONTHLY": "TIMEFRAME_MN1",
    "WEEKLY": "TIMEFRAME_W1",
    "DAILY": "TIMEFRAME_D1",
    "H4": "TIMEFRAME_H4",
    "H1": "TIMEFRAME_H1",
    "M15": "TIMEFRAME_M15",
    "M5": "TIMEFRAME_M5",
    "M3": "TIMEFRAME_M3",
    "M1": "TIMEFRAME_M1"
}

# Common Trading Parameters
RISK_REWARD_RATIO = 2.0  # 1:2 risk-to-reward ratio
MAX_RISK_PERCENT = 7.0  # Maximum risk per trade (percentage of account)

# Drawdown Management
MAX_DRAWDOWN_PERCENT = 5.0  # Maximum allowed drawdown (percentage of account)
MAX_DAILY_DRAWDOWN_PERCENT = 5.0  # Maximum allowed daily drawdown (percentage of daily starting equity)
MAX_DRAWDOWN_AMOUNT = 10000.0  # Maximum allowed drawdown in currency units
CLOSE_POSITIONS_ON_MAX_DRAWDOWN = True  # Whether to close all positions when max drawdown is reached


# NY Trading Session Times
NY_SESSION_START = time(9, 30)  # 9:30 AM EDT
NY_SESSION_END = time(16, 0)  # 4:00 PM EDT

# Default Silver Bullet Window
SILVER_BULLET_WINDOW_START = time(10, 0)  # 10:00 AM ET
SILVER_BULLET_WINDOW_END = time(11, 0)  # 11:00 AM ET
AUTO_CLOSE_MINUTES_BEFORE_CLOSE = 30  # Close trades 20 min before market close

# Global Strategy Parameters
FVG_MIN_SIZE_POINTS = 5  # Default minimum size for FVGs (overridden by instrument configs)
LIQUIDITY_SWEEP_MIN_PIPS = 3  # Minimum size for liquidity sweeps
SAFE_BUFFER_POINTS = 2  # Default buffer for stop loss (overridden by instrument configs)

# Define different trading instruments with their specific parameters
INSTRUMENTS = {
    "defaults": {
        "default_lot_size": 0.1,
        "max_lot_size": 1.0,
        "min_stop_distance": 10,
        "buffer_points": 2,
        # Other default parameters
    },
    "NAS100": {
        "symbol": "NAS100",
        "window_start": time(10, 00),  # 10:00 AM NY
        "window_end": time(11, 00),  # 11:00 AM NY
        "min_stop_distance": 10,  # in points
        "buffer_points": 5,  # buffer for SL
        "default_lot_size": 50.0,  # Default lot size
        "max_lot_size": 1000.0,  # Maximum lot size
        "max_broker_lot_size": 1000.0,
        "fvg_min_size": 10,  # Minimum FVG size in points
        "point_value": 5.0,  # Value of 1 point
        "description": "NASDAQ 100 Index",
        "trades_on_weekend": False,  # Doesn't trade on weekends
        "max_trades_per_day": 10,  # Instrument-specific setting
        "max_concurrent_trades": 5,  # Instrument-specific setting
        "alias": ["NQ_100", "US100.cash", "USTECH100", "USTEC"]  # Add alias list for symbol variations
    },
    "NQ100": {
        "symbol": "[NQ100]",
        "window_start": time(10, 00),  # 10:00 AM NY
        "window_end": time(11, 00),  # 11:00 AM NY
        "min_stop_distance": 10,  # in points
        "buffer_points": 5,  # buffer for SL
        "default_lot_size": 15.0,  # Default lot size
        "max_lot_size": 21.0,  # Maximum lot size
        "max_broker_lot_size": 50.0,
        "fvg_min_size": 10,  # Minimum FVG size in points
        "point_value": 0.01,  # Value of 1 point
        "description": "NASDAQ 100 Index",
        "trades_on_weekend": False,  # Doesn't trade on weekends
        "max_trades_per_day": 10,  # Instrument-specific setting
        "max_concurrent_trades": 5,  # Instrument-specific setting
        "alias": ["NQ100", "US100.cash", "USTECH100", "USTEC"]  # Add alias list for symbol variations
    },

    "GBPUSD": {
        "symbol": "GBPUSD",
        "window_start": time(3, 0),  # 3:00 AM NY
        "window_end": time(4, 0),  # 4:00 AM NY
        "min_stop_distance": 15,  # in pips
        "buffer_points": 5,  # buffer for SL
        "default_lot_size": 17.0,  # Default lot size
        "max_lot_size": 32.0,  # Maximum lot size
        "max_broker_lot_size": 52.0,
        "fvg_min_size": 5,  # Reduce from 5 to 3 pips for more detection
        "point_value": 0.0001,  # Value of 1 pip
        "description": "British Pound vs US Dollar",
        "trades_on_weekend": False,  # Doesn't trade on weekends
        "max_trades_per_day": 2,  # Instrument-specific setting
        "max_concurrent_trades": 1  # Instrument-specific setting
    },
    "AUDUSD": {
        "symbol": "AUDUSD",
        "windows": [  # Multiple time windows
            {"start": time(3, 0), "end": time(4, 0)},  # 3:00-4:00 AM NY
            {"start": time(19, 0), "end": time(22, 0)}  # 7:00-8:00 PM NY
        ],
        "min_stop_distance": 10,  # in pips
        "buffer_points": 5,  # buffer for SL
        "default_lot_size": 15.0,  # Default lot size
        "max_lot_size": 21.0,  # Maximum lot size
        "max_broker_lot_size": 30.0,
        "fvg_min_size": 2,  # Minimum FVG size in pips
        "point_value": 0.0001,  # Value of 1 pip
        "description": "Australian Dollar vs US Dollar",
        "trades_on_weekend": False,  # Doesn't trade on weekends
        "max_trades_per_day": 2,  # Instrument-specific setting
        "max_concurrent_trades": 1  # Instrument-specific setting
    },
    "XAUUSD": {
        "symbol": "XAUUSD",
        "window_start": time(10, 0),  # 10:00 AM NY
        "window_end": time(11, 0),  # 11:00 AM NY
        "min_stop_distance": 25,  # in pips
        "buffer_points": 5,  # buffer for SL
        "default_lot_size": 0.2,  # Default lot size
        "max_lot_size": 0.3,  # # Maximum lot size
        "max_broker_lot_size": 0.3,
        "fvg_min_size": 10,  # Minimum FVG size in pips
        "point_value": 0.01,  # Value of 1 pip for gold
        "description": "Gold vs US Dollar",
        "trades_on_weekend": False,  # Doesn't trade on weekends
        "max_trades_per_day": 2,  # Instrument-specific setting
        "max_concurrent_trades": 1  # Instrument-specific setting
    },
    "BTCUSD": {
        "symbol": "BTCUSD",
        "windows": [  # Multiple time windows to capture different trading sessions
            {"start": time(2, 0), "end": time(3, 0)},  # 2:00-3:00 AM NY (Asian session)
            {"start": time(8, 0), "end": time(9, 0)},  # 8:00-9:00 AM NY (European session)
            {"start": time(20, 0), "end": time(22, 0)}  # 2:00-3:00 PM NY (US session)
        ],
        "min_stop_distance": 15,  # Bitcoin requires wider stops due to volatility
        "buffer_points": 10,  # Buffer for SL
        "default_lot_size": 0.1,  # Start with smaller lots for Bitcoin
        "max_lot_size": 0.5,  # Maximum lot size
        "max_broker_lot_size": 0.5,
        "fvg_min_size": 5,  # Minimum FVG size in points - larger for BTC
        "point_value": 1.0,  # Each point = $1 in BTC
        "description": "Bitcoin vs US Dollar",
        "trades_on_weekend": True,  # trade on weekends
        "max_trades_per_day": 2,  # Instrument-specific setting
        "max_concurrent_trades": 1  # Instrument-specific setting
    }
}
# 'I:NDX,I:SPX,C:GBPUSD,C:AUDUSD,C:XAUUSD'
# 'C:GBPUSD, I:NDX,I:SPX,C:AUDUSD,C:XAUUSD'
# 2020,2021,2022,2023,2024


# Specify which instruments to trade (you can enable/disable instruments here)
ACTIVE_INSTRUMENTS = ["NAS100", "XAUUSD", "AUDUSD", "BTCUSD"]

# MT5 Broker Configurations
# In config.py, add to MT5_BROKERS dictionary:

MT5_BROKERS = {
    "pepperstone": {
        "name": "Pepperstone",
        "path": r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe",
        "description": "Pepperstone MetaTrader 5"
    },
    "fp_markets": {
        "name": "FP Markets",
        "path": r"C:\Program Files\FP Markets MetaTrader 5\terminal64.exe",
        "description": "FP Markets MetaTrader 5"
    },
    "admirals": {  # Add this new entry
        "name": "Admirals Group",
        "path": r"C:\Program Files\Admirals Group MT5 Terminal\terminal64.exe",
        "description": "Admirals Group MetaTrader 5"
    }
    # Additional brokers can be added here later
}

# Default broker to use (can be overridden via command line)
DEFAULT_BROKER = "pepperstone"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Debug Mode - Set to True to enable additional debugging output
DEBUG_MODE = True

# Add this to config.py
import json
import os

# File to store drawdown tracking information
DRAWDOWN_TRACKING_FILE = "drawdown_tracking.json"

# Ensure base log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create log directories for each instrument
for instrument in ACTIVE_INSTRUMENTS:
    instrument_log_dir = os.path.join(LOG_DIR, instrument)
    os.makedirs(instrument_log_dir, exist_ok=True)
