# silver_bullet_bot/core/strategy.py

import logging
import os
import json
from datetime import datetime, time, timedelta
import MetaTrader5 as mt5
import pytz
import pandas as pd

from silver_bullet_bot.config import (
    INSTRUMENTS, TIMEFRAMES, SAFE_BUFFER_POINTS,
    RISK_REWARD_RATIO, LOG_DIR, MAX_RISK_PERCENT
)
from silver_bullet_bot.core.utils import (
    copy_rates_from_pos, detect_liquidity_sweep, find_fvg,
    find_breaker_or_ob, calculate_lot_size, determine_bias,
    is_in_fibonacci_zone, calculate_fibonacci_levels
)
from silver_bullet_bot.core.timezone_utils import (
    convert_mt5_to_ny, is_silver_bullet_window,
    is_ny_trading_time, format_time_for_display
)


class SilverBulletStrategy:
    """
    ICT Silver Bullet trading strategy implementation
    """

    def __init__(self, instrument_name, symbol, symbol_mapper):
        """
        Initialize the strategy for a specific instrument

        Parameters:
        -----------
        instrument_name : str
            The standardized instrument name as defined in config
        symbol : str
            The broker-specific symbol
        symbol_mapper : SymbolMapper
            Symbol mapper instance for broker-specific symbol conversion
        """
        self.instrument_name = instrument_name
        self.symbol = symbol
        self.symbol_mapper = symbol_mapper

        # Load instrument configuration
        if instrument_name not in INSTRUMENTS:
            raise ValueError(f"Instrument {instrument_name} not found in configuration")

        self.config = INSTRUMENTS[instrument_name]

        # Get default values if not specified in instrument config
        defaults = INSTRUMENTS.get("defaults", {})
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

        # Set up logging for this instrument
        self.logger = self._setup_logger()

        # Initialize strategy state
        self.reset_state()

        # Initialize drawdown tracking
        self.drawdown_state = self._load_drawdown_state()

        ny_time = convert_mt5_to_ny(datetime.now())
        self.logger.info(
            f"Strategy initialized for {instrument_name} (Symbol: {symbol}) at {ny_time.strftime('%H:%M:%S')} NY time")

    def _setup_logger(self):
        """Set up a logger for this instrument"""
        logger = logging.getLogger(f'silver_bullet.{self.instrument_name}')

        # Create file handler for instrument-specific log
        instrument_log_dir = os.path.join(LOG_DIR, self.instrument_name)
        os.makedirs(instrument_log_dir, exist_ok=True)
        log_file = os.path.join(instrument_log_dir, f"{self.instrument_name}.log")
        file_handler = logging.FileHandler(log_file)

        # Set format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        return logger

    def reset_state(self):
        """Reset the strategy state for a new trading session"""
        self.trade_open = False
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.direction = None
        self.position_id = None
        self.entry_time = None
        self.lot_size = None
        self.trades_executed_today = 0  # Change from boolean to counter
        self.liquidity_sweep_data = None
        self.fvg_data = None
        self.breaker_data = None
        self.moved_to_breakeven = False
        self.htf_bias = None
        self.fib_levels = None

        # For tracking potential setup stages
        self.potential_setup = {
            'htf_bias': None,
            'liquidity_sweep': None,
            'displacement': None,
            'fvg': None,
            'breaker': None,
            'entry_pending': False,
            'entry_price': None,
            'stop_price': None
        }

        # Drawdown tracking (separate from daily reset)
        if not hasattr(self, 'drawdown_state'):
            self.drawdown_state = {
                'starting_equity': 0,
                'current_equity': 0,
                'max_drawdown': 0,
                'daily_starting_equity': 0,
                'daily_drawdown': 0,
                'date': datetime.now().strftime('%Y-%m-%d')
            }

        ny_time = convert_mt5_to_ny(datetime.now())
        self.logger.info(f"Strategy state reset at {ny_time.strftime('%H:%M:%S')} NY time")

    def is_trading_window_open(self, current_time_mt5):
        """
        Check if the current time is within the trading window for this instrument

        Parameters:
        -----------
        current_time_mt5 : datetime
            Current MT5 server time

        Returns:
        --------
        bool
            True if within trading window, False otherwise
        """
        # Convert to NY time
        current_time_ny = convert_mt5_to_ny(current_time_mt5)

        # Check if weekend trading is allowed for this instrument
        if not self.config.get('trades_on_weekend', False):
            if current_time_ny.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                # Only log this once every hour to avoid log spam
                if current_time_ny.minute == 0 and current_time_ny.second < 5:
                    self.logger.info(
                        f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Weekend - no trading")
                return False

        # Check if within NY trading hours
        if not is_ny_trading_time(current_time_ny):
            # Only log this once every hour to avoid log spam
            if current_time_ny.minute == 0 and current_time_ny.second < 5:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Outside NY trading hours")
            return False

        # Check if within instrument-specific window(s)
        # Some instruments have multiple windows, others have just one
        is_in_window = False
        window_info = ""

        if 'windows' in self.config:
            # Multiple windows case
            for i, window in enumerate(self.config['windows']):
                window_start = window['start']
                window_end = window['end']

                if window_start <= current_time_ny.time() <= window_end:
                    is_in_window = True
                    window_info = f"Window {i + 1}: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} NY"
                    break
                else:
                    # Show upcoming windows if not in any window
                    if not is_in_window and current_time_ny.second == 0 and current_time_ny.minute % 30 == 0:
                        # Calculate time until next window
                        hours_today = current_time_ny.time().hour + current_time_ny.time().minute / 60
                        window_start_hours = window_start.hour + window_start.minute / 60

                        # If this window is later today
                        if window_start_hours > hours_today:
                            hours_until = window_start_hours - hours_today
                            self.logger.info(
                                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Next trading window {i + 1} starts in {hours_until:.1f} hours")
        else:
            # Single window case
            window_start = self.config.get('window_start')
            window_end = self.config.get('window_end')

            if window_start and window_end:
                if window_start <= current_time_ny.time() <= window_end:
                    is_in_window = True
                    window_info = f"Window: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} NY"
                else:
                    # Show upcoming window if not in it
                    if current_time_ny.second == 0 and current_time_ny.minute % 30 == 0:
                        # Calculate time until window
                        hours_today = current_time_ny.time().hour + current_time_ny.time().minute / 60
                        window_start_hours = window_start.hour + window_start.minute / 60

                        # If window is later today
                        if window_start_hours > hours_today:
                            hours_until = window_start_hours - hours_today
                            self.logger.info(
                                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Trading window starts in {hours_until:.1f} hours")
            else:
                # Use default Silver Bullet window
                is_in_window = is_silver_bullet_window(current_time_ny)
                if is_in_window:
                    window_info = "Default Silver Bullet window"

        # Log window status changes - we need to track the previous state to avoid log spam
        if not hasattr(self, '_prev_window_state'):
            self._prev_window_state = False

        # If state changed, log it
        if is_in_window != self._prev_window_state:
            if is_in_window:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: ENTERED trading window - {window_info}")
            else:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: EXITED trading window")
            self._prev_window_state = is_in_window
        # Otherwise, periodically log the current window status
        elif is_in_window and current_time_ny.minute % 15 == 0 and current_time_ny.second < 5:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: In trading window - {window_info}")

        return is_in_window

    def determine_htf_bias(self):
        """
        Determine higher timeframe bias for the instrument

        Returns:
        --------
        str
            'bullish', 'bearish', or 'neutral'
        """
        # Get current NY time for logging
        ny_time = convert_mt5_to_ny(datetime.now())
        self.logger.info(
            f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: Analyzing higher timeframes for bias determination")

        # Get bias from individual timeframes for detailed logging
        timeframes_to_check = ['DAILY', 'H4', 'H1']
        bias_by_tf = {}

        for tf in timeframes_to_check:
            # Get data for this timeframe
            df = copy_rates_from_pos(self.symbol, tf, 0, 10)

            if df is None or len(df) < 5:
                self.logger.warning(
                    f"[{ny_time.strftime('%H:%M:%S')} NY] Not enough data for {tf} bias determination, skipping")
                bias_by_tf[tf] = 'neutral'
                continue

            # Log the last 3 candles for analysis
            self.logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: {tf} last 3 candles:")
            for i in range(min(3, len(df))):
                candle = df.iloc[i]
                self.logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: {tf} Candle {i} - "
                                 f"O: {candle['open']:.5f}, H: {candle['high']:.5f}, L: {candle['low']:.5f}, C: {candle['close']:.5f}, "
                                 f"Time: {candle['time']}")

            # Simple bias determination based on last N candles
            # Check if making higher highs and higher lows (bullish)
            higher_highs = df['high'].iloc[0] > df['high'].iloc[1] > df['high'].iloc[2]
            higher_lows = df['low'].iloc[0] > df['low'].iloc[1] > df['low'].iloc[2]

            # Check if making lower highs and lower lows (bearish)
            lower_highs = df['high'].iloc[0] < df['high'].iloc[1] < df['high'].iloc[2]
            lower_lows = df['low'].iloc[0] < df['low'].iloc[1] < df['low'].iloc[2]

            # Log the actual conditions
            self.logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: {tf} conditions - "
                             f"Higher Highs: {higher_highs} ({df['high'].iloc[0]:.5f} > {df['high'].iloc[1]:.5f} > {df['high'].iloc[2]:.5f}), "
                             f"Higher Lows: {higher_lows} ({df['low'].iloc[0]:.5f} > {df['low'].iloc[1]:.5f} > {df['low'].iloc[2]:.5f})")
            self.logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: {tf} conditions - "
                             f"Lower Highs: {lower_highs} ({df['high'].iloc[0]:.5f} < {df['high'].iloc[1]:.5f} < {df['high'].iloc[2]:.5f}), "
                             f"Lower Lows: {lower_lows} ({df['low'].iloc[0]:.5f} < {df['low'].iloc[1]:.5f} < {df['low'].iloc[2]:.5f})")

            # Assign bias for this timeframe
            if higher_highs and higher_lows:
                bias_by_tf[tf] = 'bullish'
            elif lower_highs and lower_lows:
                bias_by_tf[tf] = 'bearish'
            else:
                bias_by_tf[tf] = 'neutral'

            self.logger.info(
                f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: {tf} bias = {bias_by_tf[tf]}")

        # Calculate overall bias
        if not bias_by_tf:
            self.logger.warning(
                f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: Could not determine bias on any timeframe")
            return 'neutral'

        # Count bullish, bearish and neutral signals
        bullish_count = sum(1 for bias in bias_by_tf.values() if bias == 'bullish')
        bearish_count = sum(1 for bias in bias_by_tf.values() if bias == 'bearish')
        neutral_count = sum(1 for bias in bias_by_tf.values() if bias == 'neutral')

        self.logger.info(
            f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: Bias summary - Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}")

        # Determine final bias (need at least 2 timeframes agreeing)
        if bullish_count >= 2:
            final_bias = 'bullish'
        elif bearish_count >= 2:
            final_bias = 'bearish'
        else:
            final_bias = 'neutral'

        # Get 4H data for Fibonacci levels
        df_4h = copy_rates_from_pos(self.symbol, 'H4', 0, 20)

        if df_4h is not None and len(df_4h) > 5:
            # Find recent swing high and low
            high = df_4h['high'].max()
            high_idx = df_4h['high'].idxmax()
            high_time = df_4h.iloc[high_idx]['time']

            low = df_4h['low'].min()
            low_idx = df_4h['low'].idxmin()
            low_time = df_4h.iloc[low_idx]['time']

            self.logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: 4H Swing points - "
                             f"High: {high:.5f} at {high_time}, Low: {low:.5f} at {low_time}")

            # Calculate Fibonacci levels
            self.fib_levels = calculate_fibonacci_levels(high, low)
            self.logger.info(
                f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: Fibonacci levels calculated:")
            for level, value in self.fib_levels.items():
                self.logger.info(
                    f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: Fib {level}%: {value:.5f}")

        self.logger.info(
            f"[{ny_time.strftime('%H:%M:%S')} NY] {self.instrument_name}: Final HTF bias determined: {final_bias}")
        return final_bias

    def check_for_liquidity_sweep(self):
        """
        Check for a liquidity sweep (stop hunt) in the 1M chart

        Returns:
        --------
        tuple or None
            (sweep_time, sweep_price, displacement_index) if sweep detected, None otherwise
        """
        current_time_ny = convert_mt5_to_ny(datetime.now())
        self.logger.info(
            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Scanning for liquidity sweep")

        # Get 1M and 5M data for sweep detection (check both timeframes)
        df_1m = copy_rates_from_pos(self.symbol, 'M1', 0, 30)  # Look back further
        df_5m = copy_rates_from_pos(self.symbol, 'M5', 0, 15)  # Check 5M too

        if df_1m is None:
            self.logger.warning(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Could not get 1M data")
            return None

        # Log recent price action for analysis
        if len(df_1m) > 0:
            recent_1m = df_1m.iloc[0]
            self.logger.debug(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Recent 1M - O: {recent_1m['open']:.5f}, H: {recent_1m['high']:.5f}, L: {recent_1m['low']:.5f}, C: {recent_1m['close']:.5f}")

        if df_5m is not None and len(df_5m) > 0:
            recent_5m = df_5m.iloc[0]
            self.logger.debug(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Recent 5M - O: {recent_5m['open']:.5f}, H: {recent_5m['high']:.5f}, L: {recent_5m['low']:.5f}, C: {recent_5m['close']:.5f}")

        # Based on HTF bias, look for specific sweep direction
        direction = 'buy' if self.htf_bias == 'bullish' else 'sell'

        # First try 1M timeframe
        self.logger.debug(
            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Checking 1M for {direction} liquidity sweep")
        sweep_data_1m = detect_liquidity_sweep(df_1m, direction)

        # Then try 5M timeframe if 1M didn't find anything
        sweep_data_5m = None
        if df_5m is not None and sweep_data_1m is None:
            self.logger.debug(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Checking 5M for {direction} liquidity sweep")
            sweep_data_5m = detect_liquidity_sweep(df_5m, direction)

        # Use whichever timeframe found a sweep, prioritize 1M
        sweep_data = sweep_data_1m or sweep_data_5m

        if sweep_data:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: {direction.upper()} liquidity sweep detected at {sweep_data[0]}, price: {sweep_data[1]:.5f}")

            # Update potential setup
            self.potential_setup['liquidity_sweep'] = {
                'time': sweep_data[0],
                'price': sweep_data[1],
                'displacement_index': sweep_data[2]
            }

            return sweep_data
        else:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: No valid {direction} liquidity sweep detected")
            return None

    def check_for_entry_setup(self):
        """
        Check for a valid entry setup (FVG or Breaker/OB)

        Returns:
        --------
        bool
            True if valid setup found, False otherwise
        """
        current_time_ny = convert_mt5_to_ny(datetime.now())
        self.logger.info(
            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Checking for entry setup")

        # We need a liquidity sweep first
        if not self.potential_setup['liquidity_sweep']:
            self.logger.debug(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: No liquidity sweep detected yet")
            return False

        # Direction based on HTF bias
        direction = 'buy' if self.htf_bias == 'bullish' else 'sell'

        # Get 1M data for FVG detection
        df_1m = copy_rates_from_pos(self.symbol, 'M1', 0, 15)

        if df_1m is None:
            self.logger.warning(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Could not get 1M data")
            return False

        # Log recent candles for analysis
        for i in range(min(3, len(df_1m))):
            candle = df_1m.iloc[i]
            self.logger.debug(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: 1M candle {i} - O: {candle['open']:.5f}, H: {candle['high']:.5f}, L: {candle['low']:.5f}, C: {candle['close']:.5f}")

        # First, look for FVG
        fvg_min_size = self.config.get('fvg_min_size', 5)
        self.logger.info(
            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Looking for {direction} FVG with min size {fvg_min_size}")
        fvg = find_fvg(df_1m, direction, fvg_min_size)

        if fvg:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Found {direction} FVG - Range: {fvg['low']:.5f} - {fvg['high']:.5f}, Size: {fvg['size']:.5f}")

            # Check if FVG is in the correct Fibonacci zone
            if self.fib_levels and is_in_fibonacci_zone(fvg['midpoint'], self.fib_levels, direction):
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: FVG at {fvg['midpoint']:.5f} is in correct Fibonacci zone")

                # Update potential setup
                self.potential_setup['fvg'] = fvg
                self.potential_setup['entry_price'] = fvg['midpoint']
                self.potential_setup['entry_pending'] = True

                # Calculate stop loss based on liquidity sweep
                sweep_price = self.potential_setup['liquidity_sweep']['price']
                buffer = self.config.get('buffer_points', SAFE_BUFFER_POINTS) * 0.0001  # Convert to price

                if direction == 'buy':
                    self.potential_setup['stop_price'] = sweep_price - buffer
                else:
                    self.potential_setup['stop_price'] = sweep_price + buffer

                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Entry setup ready - {direction.upper()} at {fvg['midpoint']:.5f}, "
                    f"stop at {self.potential_setup['stop_price']:.5f}, buffer: {buffer:.5f}")
                return True
            else:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: FVG at {fvg['midpoint']:.5f} is NOT in correct Fibonacci zone, skipping")
                # Log Fibonacci levels for analysis
                if self.fib_levels:
                    self.logger.debug(
                        f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Fibonacci levels - 0%: {self.fib_levels['0.0']:.5f}, 50%: {self.fib_levels['50.0']:.5f}, 100%: {self.fib_levels['100.0']:.5f}")
        else:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: No valid {direction} FVG found, checking for breaker/order block")

        # If no valid FVG, try to find a Breaker or Order Block
        if not self.potential_setup.get('fvg'):
            displacement_index = self.potential_setup['liquidity_sweep']['displacement_index']
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Looking for {direction} breaker/order block")
            breaker = find_breaker_or_ob(df_1m, direction, displacement_index)

            if breaker:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Found {direction} breaker/block - Range: {breaker['low']:.5f} - {breaker['high']:.5f}")

                # Update potential setup
                self.potential_setup['breaker'] = breaker
                self.potential_setup['entry_price'] = breaker['retest_level']
                self.potential_setup['entry_pending'] = True

                # Calculate stop loss based on liquidity sweep
                sweep_price = self.potential_setup['liquidity_sweep']['price']
                buffer = self.config.get('buffer_points', SAFE_BUFFER_POINTS) * 0.0001  # Convert to price

                if direction == 'buy':
                    self.potential_setup['stop_price'] = sweep_price - buffer
                else:
                    self.potential_setup['stop_price'] = sweep_price + buffer

                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Entry setup ready - {direction.upper()} at {breaker['retest_level']:.5f}, "
                    f"stop at {self.potential_setup['stop_price']:.5f}, buffer: {buffer:.5f}")
                return True
            else:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: No valid {direction} breaker/order block found")

        self.logger.info(
            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: No valid entry setup found")
        return False

    def check_entry_conditions(self):
        """
        Check if current price has reached entry level

        Returns:
        --------
        bool
            True if entry conditions met, False otherwise
        """
        # We need a pending entry setup
        if not self.potential_setup['entry_pending']:
            return False

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            self.logger.warning(f"Could not get current price for {self.symbol}")
            return False

        current_bid = tick.bid
        current_ask = tick.ask

        direction = 'buy' if self.htf_bias == 'bullish' else 'sell'
        entry_price = self.potential_setup['entry_price']

        # For buy entry, price should drop to or below entry price
        if direction == 'buy' and current_bid <= entry_price:
            self.logger.info(f"Buy entry condition met at {current_bid} (entry price: {entry_price})")
            return True

        # For sell entry, price should rise to or above entry price
        if direction == 'sell' and current_ask >= entry_price:
            self.logger.info(f"Sell entry condition met at {current_ask} (entry price: {entry_price})")
            return True

        return False

    def execute_trade(self):
        """
        Execute a trade based on the current setup

        Returns:
        --------
        bool
            True if trade executed successfully, False otherwise
        """
        self.logger.info(f"Executing trade for {self.symbol}")

        # We need entry and stop prices
        if not (self.potential_setup['entry_price'] and self.potential_setup['stop_price']):
            self.logger.warning("Cannot execute trade: missing entry or stop price")
            return False

        # Get account info for lot sizing
        account_info = mt5.account_info()

        if account_info is None:
            self.logger.error(f"Could not get account info: {mt5.last_error()}")
            return False

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            self.logger.warning(f"Could not get current price for {self.symbol}")
            return False

        direction = 'buy' if self.htf_bias == 'bullish' else 'sell'
        entry_price = self.potential_setup['entry_price']
        stop_price = self.potential_setup['stop_price']

        # Adjust entry price based on current price and direction
        if direction == 'buy':
            entry_price = tick.ask
        else:
            entry_price = tick.bid

        # Get current open positions for this instrument
        positions = mt5.positions_get(symbol=self.symbol)
        open_positions_lot_sum = sum(position.volume for position in positions) if positions else 0

        # Calculate lot size with open positions consideration
        lot_size = calculate_lot_size(
            account_info.balance,
            stop_price,
            entry_price,
            self.config,
            open_positions_lot_sum
        )

        # Round lot size to broker's acceptable value
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is not None:
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step

        # Prepare trade request
        trade_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL

        # Calculate take profit (if using fixed RR)
        take_profit = None
        if RISK_REWARD_RATIO > 0:
            tp_distance = abs(entry_price - stop_price) * RISK_REWARD_RATIO
            if direction == 'buy':
                take_profit = entry_price + tp_distance
            else:
                take_profit = entry_price - tp_distance

        # Create the basic trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": entry_price,
            "sl": stop_price,
            "tp": take_profit,
            "deviation": 10,  # Allow slippage
            "magic": 123456,  # Magic number to identify bot trades
            "comment": f"Silver Bullet {self.instrument_name}",
            "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
        }

        # Get symbol filling modes
        if symbol_info is not None:
            # Check the filling mode flags
            filling_modes = symbol_info.filling_mode
            self.logger.info(f"Symbol {self.symbol} supported filling modes: {filling_modes}")

            # Try to set the most appropriate filling mode
            if filling_modes & mt5.SYMBOL_FILLING_FOK:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif filling_modes & mt5.SYMBOL_FILLING_IOC:
                request["type_filling"] = mt5.ORDER_FILLING_IOC
            else:
                # If neither FOK nor IOC supported, don't specify (use default)
                self.logger.info(f"Using default filling mode for {self.symbol}")
        else:
            self.logger.warning(f"Could not get symbol info for {self.symbol}")

        # Send the trade request
        result = mt5.order_send(request)

        if result is None:
            self.logger.error(f"Trade execution failed: {mt5.last_error()}")
            return False

        # Handle result
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Trade executed successfully: {direction} {lot_size} lot(s) at {entry_price}, "
                             f"SL: {stop_price}, TP: {take_profit}")

            # Update trade state
            self.trade_open = True
            self.entry_price = entry_price
            self.stop_loss = stop_price
            self.take_profit = take_profit
            self.direction = direction
            self.position_id = result.order
            self.entry_time = datetime.now()
            self.lot_size = lot_size
            self.trades_executed_today += 1  # Increment counter instead of setting flag

            # Log trade details
            self._log_trade("OPEN", {
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_price,
                "take_profit": take_profit,
                "lot_size": lot_size,
                "position_id": result.order,
                "time": datetime.now().isoformat()
            })

            return True
        else:
            self.logger.error(f"Trade execution failed with code {result.retcode}: {result.comment}")
            return False

    def manage_open_trade(self):
        """
        Manage an open trade (check for breakeven, etc.)
        """
        if not self.trade_open:
            return

        # Get current position
        position = mt5.positions_get(symbol=self.symbol)

        if not position:
            self.logger.warning(f"No open position found for {self.symbol}, resetting state")
            self.trade_open = False
            return

        position = position[0]._asdict()

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            self.logger.warning(f"Could not get current price for {self.symbol}")
            return

        current_price = tick.bid if self.direction == 'buy' else tick.ask

        # Check if we need to move stop loss to breakeven
        if not self.moved_to_breakeven:
            # Calculate 1R distance
            r_distance = abs(self.entry_price - self.stop_loss)

            # For a buy trade, check if price has moved up by 1R
            if self.direction == 'buy' and current_price >= self.entry_price + r_distance:
                self._move_stop_to_breakeven()

            # For a sell trade, check if price has moved down by 1R
            elif self.direction == 'sell' and current_price <= self.entry_price - r_distance:
                self._move_stop_to_breakeven()

    def _move_stop_to_breakeven(self):
        """Move stop loss to breakeven"""
        if not self.trade_open:
            return

        self.logger.info(f"Moving stop loss to breakeven at {self.entry_price}")

        # Create the modify request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "sl": self.entry_price,
            "position": self.position_id
        }

        # Send the modify request
        result = mt5.order_send(request)

        if result is None:
            self.logger.error(f"Failed to move stop to breakeven: {mt5.last_error()}")
            return

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info("Stop loss moved to breakeven successfully")
            self.stop_loss = self.entry_price
            self.moved_to_breakeven = True

            # Log trade adjustment
            self._log_trade("ADJUST", {
                "action": "move_to_breakeven",
                "new_stop_loss": self.entry_price,
                "time": datetime.now().isoformat()
            })
        else:
            self.logger.error(f"Failed to move stop to breakeven with code {result.retcode}: {result.comment}")

    def check_exit_conditions(self, current_time_mt5):
        """
        Check if it's time to exit the trade

        Parameters:
        -----------
        current_time_mt5 : datetime
            Current MT5 server time

        Returns:
        --------
        bool
            True if exit conditions met, False otherwise
        """
        if not self.trade_open:
            return False

        # Convert to NY time
        current_time_ny = convert_mt5_to_ny(current_time_mt5)

        # Calculate auto-close time based on market close time
        from silver_bullet_bot.config import NY_SESSION_END, AUTO_CLOSE_MINUTES_BEFORE_CLOSE

        # Create auto-close time by subtracting minutes from market close
        auto_close_hour = NY_SESSION_END.hour
        auto_close_minute = NY_SESSION_END.minute - AUTO_CLOSE_MINUTES_BEFORE_CLOSE

        # Adjust for negative minutes
        if auto_close_minute < 0:
            auto_close_hour -= 1
            auto_close_minute += 60

        auto_close_time = time(auto_close_hour, auto_close_minute)

        # Check if it's time for market close auto-exit
        if current_time_ny.time() >= auto_close_time:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] Auto-close exit triggered for {self.instrument_name} ({AUTO_CLOSE_MINUTES_BEFORE_CLOSE} minutes before market close)")
            return True

        # You can also keep the original Silver Bullet time-based exit (10:35 AM NY time)
        sb_exit_time = time(10, 35)  # 10:35 AM

        if current_time_ny.time() >= sb_exit_time:
            self.logger.info(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] Silver Bullet time-based exit triggered for {self.instrument_name}")
            return True

        # Log time remaining until exit (for debugging purposes)
        minutes_until_exit = min(
            ((sb_exit_time.hour - current_time_ny.time().hour) * 60 + (
                        sb_exit_time.minute - current_time_ny.time().minute)),
            ((auto_close_time.hour - current_time_ny.time().hour) * 60 + (
                        auto_close_time.minute - current_time_ny.time().minute))
        )

        if minutes_until_exit > 0 and minutes_until_exit % 5 == 0:  # Log every 5 minutes
            self.logger.debug(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {minutes_until_exit} minutes until earliest time-based exit")

        return False

    def close_trade(self):
        """
        Close the current open trade

        Returns:
        --------
        bool
            True if trade closed successfully, False otherwise
        """
        if not self.trade_open:
            self.logger.warning("No open trade to close")
            return False

        ny_time = convert_mt5_to_ny(datetime.now())
        self.logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] Closing trade for {self.symbol}")

        # Get current position
        position = mt5.positions_get(symbol=self.symbol)

        if not position:
            self.logger.warning(f"No open position found for {self.symbol}, resetting state")
            self.trade_open = False
            return False

        position = position[0]._asdict()

        # Prepare close request
        trade_type = mt5.ORDER_TYPE_SELL if self.direction == 'buy' else mt5.ORDER_TYPE_BUY

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            self.logger.warning(f"Could not get current price for {self.symbol}")
            return False

        close_price = tick.bid if self.direction == 'buy' else tick.ask

        # Create the close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": trade_type,
            "position": self.position_id,
            "price": close_price,
            "deviation": 10,  # Allow slippage
            "magic": 123456,  # Magic number to identify bot trades
            "comment": f"Close Silver Bullet {self.instrument_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }

        # Send the close request
        result = mt5.order_send(request)

        if result is None:
            self.logger.error(f"Trade close failed: {mt5.last_error()}")
            return False

        # Handle result
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Calculate profit
            if self.direction == 'buy':
                profit_points = close_price - self.entry_price
            else:
                profit_points = self.entry_price - close_price

            profit_r = profit_points / abs(self.entry_price - self.stop_loss)

            self.logger.info(
                f"[{ny_time.strftime('%H:%M:%S')} NY] Trade closed successfully at {close_price}, profit: {profit_points:.5f} points ({profit_r:.2f}R)")

            # Reset trade state
            self.trade_open = False
            # Reset potential setup to allow new trades
            self.potential_setup = {
                'htf_bias': self.htf_bias,  # Keep the same bias
                'liquidity_sweep': None,
                'displacement': None,
                'fvg': None,
                'breaker': None,
                'entry_pending': False,
                'entry_price': None,
                'stop_price': None
            }

            # Log trade details
            self._log_trade("CLOSE", {
                "close_price": close_price,
                "profit_points": profit_points,
                "profit_r": profit_r,
                "time": datetime.now().isoformat()
            })

            # Update drawdown state after trade close (with forced logging)
            self.update_drawdown_state(force_log=True)

            return True
        else:
            self.logger.error(f"Trade close failed with code {result.retcode}: {result.comment}")
            return False

    def _log_trade(self, action, details):
        """
        Log trade details to JSON file

        Parameters:
        -----------
        action : str
            'OPEN', 'CLOSE', or 'ADJUST'
        details : dict
            Trade details to log
        """
        sessions_dir = os.path.join(os.path.dirname(LOG_DIR), "sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(sessions_dir, f"trades_{date_str}.json")

        # Get current drawdown information
        drawdown_info = {
            "total_drawdown_pct": 0,
            "daily_drawdown_pct": 0,
            "max_drawdown_pct": 0
        }

        if hasattr(self, 'drawdown_state'):
            account_info = mt5.account_info()
            if account_info:
                current_equity = account_info.equity
                if self.drawdown_state['starting_equity'] > 0:
                    drawdown_info["total_drawdown_pct"] = (1 - current_equity / self.drawdown_state[
                        'starting_equity']) * 100

                if self.drawdown_state['daily_starting_equity'] > 0:
                    drawdown_info["daily_drawdown_pct"] = (1 - current_equity / self.drawdown_state[
                        'daily_starting_equity']) * 100

                drawdown_info["max_drawdown_pct"] = self.drawdown_state['max_drawdown']

        # Add NY time to details
        ny_time = convert_mt5_to_ny(datetime.now())
        if 'time' in details:
            details['mt5_time'] = details['time']
        details['ny_time'] = ny_time.strftime('%Y-%m-%d %H:%M:%S')

        # Create a trade log entry
        trade_log = {
            "action": action,
            "instrument": self.instrument_name,
            "symbol": self.symbol,
            "details": details,
            "drawdown": drawdown_info
        }

        # Load existing logs if file exists
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        else:
            logs = []

        # Append new log
        logs.append(trade_log)

        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        # Log to console with NY time
        self.logger.info(
            f"[{ny_time.strftime('%H:%M:%S')} NY] Trade {action}: {self.instrument_name} ({self.symbol}) - {details.get('direction', '')}")

    def run_iteration(self, current_time_mt5):
        """
        Run one iteration of the strategy

        Parameters:
        -----------
        current_time_mt5 : datetime
            Current MT5 server time
        """
        # Convert to NY time for logging
        current_time_ny = convert_mt5_to_ny(current_time_mt5)

        max_trades = self.config.get('max_trades_per_day', 1)
        if self.trades_executed_today >= max_trades:
            # Every 10 seconds log the trade status
            if current_time_ny.second % 10 == 0:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Maximum trades for today reached ({self.trades_executed_today}/{max_trades}), monitoring only")

            # But still manage open trade
            if self.trade_open:
                self.manage_open_trade()

                # Check for time-based exit
                if self.check_exit_conditions(current_time_mt5):
                    self.close_trade()
            return

        # Heartbeat log (every 10 seconds)
        if current_time_ny.second % 10 == 0:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                self.logger.debug(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Heartbeat - Bid: {tick.bid:.5f}, Ask: {tick.ask:.5f}")

        # Check drawdown limits first
        if not self.update_drawdown_state():
            self.logger.warning(
                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Drawdown limits exceeded, no new trades allowed")

            # If we have an open trade, we should still manage it
            if self.trade_open:
                self.logger.debug(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Managing open trade despite drawdown limits")
                self.manage_open_trade()

                # Check for time-based exit
                if self.check_exit_conditions(current_time_mt5):
                    self.close_trade()
            return

        # Check if we're in a trading window
        if not self.is_trading_window_open(current_time_mt5):
            # Outside trading window, can still manage open trades
            if self.trade_open:
                self.logger.debug(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Outside trading window but managing open trade")
                self.manage_open_trade()

                # Check for time-based exit
                if self.check_exit_conditions(current_time_mt5):
                    self.close_trade()
            return

        # Skip if already traded today
        if self.trades_executed_today >= self.config.get('max_trades_per_day', 1):
            # Every 10 seconds log the trade status
            # if current_time_ny.second % 10 == 0:
            #     self.logger.info(
            #         f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Maximum trades for today reached ({self.trades_executed_today}/{self.config.get('max_trades_per_day', 1)}), monitoring only")

            # But still manage open trade
            if self.trade_open:
                self.manage_open_trade()

                # Check for time-based exit
                if self.check_exit_conditions(current_time_mt5):
                    self.close_trade()
            return

        # If no trade is open, look for setup
        if not self.trade_open:
            # Step 1: Determine HTF bias if not already done
            if self.htf_bias is None:
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Starting bias determination")
                self.htf_bias = self.determine_htf_bias()
                self.potential_setup['htf_bias'] = self.htf_bias

                # Log timeframe status after determining bias
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Timeframe analysis complete - Bias: {self.htf_bias}")

            # Only proceed if we have a clear bias
            if self.htf_bias in ['bullish', 'bearish']:
                # Step 2: Check for liquidity sweep - do this on regular intervals
                if not self.potential_setup['liquidity_sweep'] and current_time_ny.second % 10 == 0:
                    self.logger.info(
                        f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Looking for {self.htf_bias} liquidity sweep")
                    self.check_for_liquidity_sweep()

                    if self.potential_setup['liquidity_sweep']:
                        sweep_time = self.potential_setup['liquidity_sweep']['time']
                        sweep_price = self.potential_setup['liquidity_sweep']['price']
                        self.logger.info(
                            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Found liquidity sweep at {sweep_time} price {sweep_price:.5f}")

                # Step 3: If we have a liquidity sweep, check for entry setup - do this frequently
                if self.potential_setup['liquidity_sweep'] and not self.potential_setup[
                    'entry_pending'] and current_time_ny.second % 5 == 0:
                    self.logger.info(
                        f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Looking for entry setup after liquidity sweep")
                    found_setup = self.check_for_entry_setup()

                    if found_setup:
                        if self.potential_setup.get('fvg'):
                            fvg = self.potential_setup.get('fvg')
                            self.logger.info(
                                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Found FVG entry setup - Range: {fvg['low']:.5f}-{fvg['high']:.5f}, Entry: {self.potential_setup['entry_price']:.5f}")
                        elif self.potential_setup.get('breaker'):
                            breaker = self.potential_setup.get('breaker')
                            self.logger.info(
                                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Found Breaker/Order Block entry setup - Range: {breaker['low']:.5f}-{breaker['high']:.5f}, Entry: {self.potential_setup['entry_price']:.5f}")

                # Step 4: If we have a pending entry, check if price has reached entry level - continuously
                if self.potential_setup['entry_pending']:
                    # Get current price
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick:
                        current_price = tick.bid if self.htf_bias == 'bullish' else tick.ask
                        price_distance = abs(current_price - self.potential_setup['entry_price'])

                        # Log entry status every 5 seconds
                        if current_time_ny.second % 5 == 0:
                            self.logger.info(
                                f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Waiting for price to reach entry at {self.potential_setup['entry_price']:.5f}, current: {current_price:.5f} (distance: {price_distance:.5f})")

                    if self.check_entry_conditions():
                        self.logger.info(
                            f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Entry conditions met, executing trade")
                        self.execute_trade()
            else:
                # Log neutral bias status every 10 seconds
                if current_time_ny.second % 10 == 0:
                    self.logger.info(
                        f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: No clear bias ({self.htf_bias}), waiting")

        # If trade is open, manage it
        if self.trade_open:
            # Monitor price vs entry/stop - log every 5 seconds
            if current_time_ny.second % 5 == 0:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    current_price = tick.bid if self.direction == 'buy' else tick.ask
                    entry_distance = current_price - self.entry_price if self.direction == 'buy' else self.entry_price - current_price
                    stop_distance = self.entry_price - self.stop_loss if self.direction == 'buy' else self.stop_loss - self.entry_price

                    # Calculate R multiple
                    r_multiple = entry_distance / stop_distance if stop_distance > 0 else 0

                    self.logger.info(
                        f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: {self.direction.upper()} trade - Current: {current_price:.5f}, Entry: {self.entry_price:.5f}, SL: {self.stop_loss:.5f}, P/L: {entry_distance:.5f} ({r_multiple:.2f}R)")

            self.manage_open_trade()

            # Check for time-based exit
            if self.check_exit_conditions(current_time_mt5):
                self.logger.info(
                    f"[{current_time_ny.strftime('%H:%M:%S')} NY] {self.instrument_name}: Exit conditions met, closing trade")
                self.close_trade()

    def _load_drawdown_state(self):
        """
        Load drawdown tracking state from file

        Returns:
        --------
        dict
            Drawdown state
        """
        from silver_bullet_bot.config import DRAWDOWN_TRACKING_FILE
        import os
        import json

        default_state = {
            'starting_equity': 0,
            'current_equity': 0,
            'max_drawdown': 0,
            'daily_starting_equity': 0,
            'daily_drawdown': 0,
            'date': datetime.now().strftime('%Y-%m-%d')
        }

        # If file doesn't exist, return default state
        if not os.path.exists(DRAWDOWN_TRACKING_FILE):
            # Get current account equity
            account_info = mt5.account_info()
            if account_info:
                equity = account_info.equity
                default_state['starting_equity'] = equity
                default_state['current_equity'] = equity
                default_state['daily_starting_equity'] = equity

            return default_state

        # Load from file
        try:
            with open(DRAWDOWN_TRACKING_FILE, 'r') as f:
                state = json.load(f)

            # Check if it's a new day, reset daily drawdown
            if state['date'] != datetime.now().strftime('%Y-%m-%d'):
                account_info = mt5.account_info()
                if account_info:
                    equity = account_info.equity
                    state['daily_starting_equity'] = equity
                    state['daily_drawdown'] = 0
                    state['date'] = datetime.now().strftime('%Y-%m-%d')

            return state
        except (json.JSONDecodeError, KeyError):
            return default_state

    def _save_drawdown_state(self):
        """Save drawdown tracking state to file"""
        from silver_bullet_bot.config import DRAWDOWN_TRACKING_FILE
        import json

        with open(DRAWDOWN_TRACKING_FILE, 'w') as f:
            json.dump(self.drawdown_state, f, indent=2)

    def update_drawdown_state(self, force_log=False):
        """
        Update drawdown tracking based on current equity

        Parameters:
        -----------
        force_log : bool
            Force logging even if not at regular interval

        Returns:
        --------
        bool
            True if drawdown limits are not exceeded, False otherwise
        """
        from silver_bullet_bot.config import MAX_DRAWDOWN_PERCENT, MAX_DAILY_DRAWDOWN_PERCENT

        account_info = mt5.account_info()
        if not account_info:
            self.logger.warning("Could not get account info for drawdown tracking")
            return True

        current_equity = account_info.equity

        # Initialize if needed
        if self.drawdown_state['starting_equity'] == 0:
            self.drawdown_state['starting_equity'] = current_equity

        if self.drawdown_state['daily_starting_equity'] == 0:
            self.drawdown_state['daily_starting_equity'] = current_equity

        # Update current equity
        self.drawdown_state['current_equity'] = current_equity

        # When calculating total drawdown - use balance instead of equity
        total_drawdown_pct = (1 - account_info.balance / self.drawdown_state['starting_equity']) * 100
        # Use balance instead of equity to only account for closed trades
        daily_drawdown_pct = (1 - account_info.balance / self.drawdown_state['daily_starting_equity']) * 100

        # Update max drawdown if needed
        if total_drawdown_pct > self.drawdown_state['max_drawdown']:
            self.drawdown_state['max_drawdown'] = total_drawdown_pct

        # Update daily drawdown
        self.drawdown_state['daily_drawdown'] = daily_drawdown_pct

        # Get current NY time
        ny_time = convert_mt5_to_ny(datetime.now())

        # Log drawdown status only when:
        # 1. At end of NY session (6pm)
        # 2. When a trade is closed (force_log=True)
        # 3. Every 30 minutes (for minimal monitoring)
        should_log = (
                (ny_time.hour == 18 and ny_time.minute == 0) or  # 6:00 PM NY
                force_log or  # After trade close
                (ny_time.minute % 30 == 0 and ny_time.second < 5)  # Every 30 minutes
        )

        if should_log:
            self.logger.info(
                f"[{ny_time.strftime('%H:%M:%S')} NY] Drawdown status - Daily: {daily_drawdown_pct:.2f}%, Total: {total_drawdown_pct:.2f}%, Max: {self.drawdown_state['max_drawdown']:.2f}%")
        else:
            # Log only at DEBUG level for other times
            self.logger.debug(
                f"Drawdown status - Daily: {daily_drawdown_pct:.2f}%, Total: {total_drawdown_pct:.2f}%, Max: {self.drawdown_state['max_drawdown']:.2f}%")

        # Check if drawdown limits are exceeded
        if total_drawdown_pct > MAX_DRAWDOWN_PERCENT:
            self.logger.warning(f"Maximum drawdown exceeded: {total_drawdown_pct:.2f}% > {MAX_DRAWDOWN_PERCENT}%")
            return False

        if daily_drawdown_pct > MAX_DAILY_DRAWDOWN_PERCENT:
            self.logger.warning(
                f"Maximum daily drawdown exceeded: {daily_drawdown_pct:.2f}% > {MAX_DAILY_DRAWDOWN_PERCENT}%")
            return False

        # If it's 6pm NY, save the state for daily record keeping
        if ny_time.hour == 18 and ny_time.minute == 0 and ny_time.second < 5:
            self.logger.info(
                f"[{ny_time.strftime('%H:%M:%S')} NY] End of session - Final daily drawdown: {daily_drawdown_pct:.2f}%, Overall drawdown: {total_drawdown_pct:.2f}%")
            self._save_drawdown_state()

        return True

    def cleanup(self):
        """Clean up resources"""
        # Close any open trade
        if self.trade_open:
            ny_time = convert_mt5_to_ny(datetime.now())
            self.logger.info(f"Cleaning up at {ny_time.strftime('%H:%M:%S')} NY time - closing open trade")
            self.close_trade()

        # Save drawdown state
        self._save_drawdown_state()