# silver_bullet_bot/backtesting/backtest_engine.py

import os
import logging
import pandas as pd
import numpy as np
import time as time_module  # Rename the time module import
from datetime import datetime, time, timedelta  # Add explicit time class import
import pytz
import json
import time
from tqdm import tqdm

# Import utilities and strategy components
from silver_bullet_bot.core.utils import (
    detect_liquidity_sweep, find_fvg, find_breaker_or_ob,
    calculate_lot_size, determine_bias, is_in_fibonacci_zone,
    calculate_fibonacci_levels
)
from silver_bullet_bot.core.timezone_utils import (
    convert_mt5_to_ny, is_silver_bullet_window, is_ny_trading_time, convert_utc_timestamp_to_ny
)
from silver_bullet_bot.config import INSTRUMENTS, MAX_RISK_PERCENT

from backtesting.data_loader import DataLoader


class BacktestEngine:
    """
    Engine for backtesting the ICT Silver Bullet strategy on historical data
    """

    def __init__(self, output_dir='backtest_results', initial_balance=10000, logger=None):
        """
        Initialize the backtest engine

        Parameters:
        -----------
        output_dir : str
            Directory for storing backtest results
        initial_balance : float
            Initial account balance for simulation
        logger : logging.Logger
            Logger instance
        """
        self.output_dir = output_dir
        self.initial_balance = initial_balance
        self.logger = logger or logging.getLogger('backtest')
        self.data_loader = DataLoader(logger=self.logger)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def run_backtest(self, instrument_name, start_date, end_date, balance=None):
        """
        Run a backtest for a specific instrument and date range

        Parameters:
        -----------
        instrument_name : str
            Instrument name as defined in config
        start_date : str or datetime
            Start date for backtest
        end_date : str or datetime
            End date for backtest
        balance : float, optional
            Starting balance (defaults to initial_balance)

        Returns:
        --------
        dict
            Backtest results
        """
        start_time = time.time()  # Track execution time

        # Load instrument configuration
        if instrument_name not in INSTRUMENTS:
            self.logger.error(f"Instrument {instrument_name} not found in configuration")
            return None

        config = INSTRUMENTS[instrument_name]

        # Set initial balance
        if balance is None:
            balance = self.initial_balance

        # Define required timeframes
        required_timeframes = ["DAILY", "H4", "H1", "M15", "M5", "M3", "M1"]

        # Load historical data for all required timeframes
        self.logger.info(f"Loading historical data for {instrument_name} from {start_date} to {end_date}")
        data = self.data_loader.load_multi_timeframe_data(
            instrument_name,
            required_timeframes,
            start_date,
            end_date
        )

        # Check if we have the necessary data
        if "M1" not in data or data["M1"].empty:
            self.logger.error(f"No 1-minute data available for {instrument_name}, cannot run backtest")
            return None

        # Create backtest state
        state = {
            'instrument': instrument_name,
            'config': config,
            'balance': balance,
            'equity': balance,
            'current_position': None,
            'trades': [],
            'equity_curve': [balance],
            'htf_bias': None,
            'fib_levels': None,
            'potential_setup': {
                'htf_bias': None,
                'liquidity_sweep': None,
                'fvg': None,
                'breaker': None,
                'entry_pending': False,
                'entry_price': None,
                'stop_price': None
            },
            'trades_by_day': {},
            'date_stats': {},
            'start_date': start_date,
            'end_date': end_date,
            'execution_time': 0
        }

        # Set up asset-specific log file
        log_file = os.path.join(self.output_dir, f"{instrument_name}_backtest.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Starting backtest for {instrument_name} from {start_date} to {end_date}")
        self.logger.info(f"Initial balance: ${balance:.2f}")

        # Process M1 data chronologically
        m1_data = data["M1"].copy()

        # Ensure time index is timezone-aware (important for session time checks)
        if m1_data['time'].dt.tz is None:
            # First localize to UTC
            utc_times = m1_data['time'].dt.tz_localize('UTC')
            # Then convert to NY time
            m1_data['time'] = utc_times.apply(lambda x: convert_utc_timestamp_to_ny(x))
            self.logger.info(f"Converted {len(m1_data)} timestamps from UTC to NY timezone")
        else:
            # Already has timezone, ensure it's NY
            m1_data['time'] = m1_data['time'].apply(lambda x: x.astimezone(pytz.timezone('America/New_York')))

        # Prepare a date-based dictionary to track trading days
        unique_dates = m1_data['time'].dt.date.unique()
        for date in unique_dates:
            state['trades_by_day'][str(date)] = []
            state['date_stats'][str(date)] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0,
                'starting_balance': state['balance']
            }

        total_bars = len(m1_data)
        self.logger.info(f"Processing {total_bars} bars of 1-minute data")

        # ADD THIS - Track whether we've already processed this bar
        last_processed_time = None

        # Process each M1 bar
        for i in tqdm(range(len(m1_data)), desc=f"Backtesting {instrument_name}"):
            # Current bar row
            row = m1_data.iloc[i]
            current_time = row['time']
            current_date = current_time.date()

            # ADD THIS - Skip if we already processed a bar with this exact timestamp
            if last_processed_time is not None and current_time == last_processed_time:
                self.logger.warning(f"Skipping duplicate bar at {current_time}")
                continue

            last_processed_time = current_time

            # Convert to NY time (for session checks)
            # Ensure it's in NY timezone
            if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
                current_time_ny = current_time.astimezone(pytz.timezone('America/New_York'))
            else:
                current_time_ny = convert_utc_timestamp_to_ny(current_time)
            # self.logger.info(
            #     f"current_time_ny: {current_time_ny}, Open: {row['open']:.5f}, High: {row['high']:.5f}, Low: {row['low']:.5f}, Close: {row['close']:.5f}")

            # Check for day change - update stats and reset daily flags
            if i > 0 and current_date != m1_data.iloc[i - 1]['time'].date():
                prev_date = m1_data.iloc[i - 1]['time'].date()

                # Update stats for previous day
                state['date_stats'][str(prev_date)]['ending_balance'] = state['balance']
                state['date_stats'][str(prev_date)]['daily_pnl'] = (
                        state['balance'] - state['date_stats'][str(prev_date)]['starting_balance']
                )

                # Reset for new day
                state['potential_setup'] = {
                    'htf_bias': None,
                    'liquidity_sweep': None,
                    'fvg': None,
                    'breaker': None,
                    'entry_pending': False,
                    'entry_price': None,
                    'stop_price': None
                }
                state['htf_bias'] = None
                state['fib_levels'] = None

                # Set starting balance for new day
                state['date_stats'][str(current_date)]['starting_balance'] = state['balance']

                self.logger.info(f"=== New trading day: {current_date} ===")

            # Process the bar
            self._process_bar(row, current_time_ny, i, data, state)

            # Update equity curve
            state['equity_curve'].append(state['equity'])

            # Close any open positions at the end of the simulation
        if state['current_position']:
            last_row = m1_data.iloc[-1]
            self.logger.info(f"End of simulation - Closing open position at {last_row['close']:.5f}")
            self._close_position(last_row, state, last_row['close'], 'end_of_simulation')

        # Calculate performance metrics
        self._calculate_performance_metrics(state)

        # Execution time
        execution_time = time.time() - start_time
        state['execution_time'] = execution_time

        self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        self.logger.info(f"Final balance: ${state['balance']:.2f}")
        self.logger.info(f"Total trades: {len(state['trades'])}")

        # Remove file handler to avoid duplicate logs
        self.logger.removeHandler(file_handler)

        # Return results
        return state

    def _process_bar(self, row, current_time_ny, bar_index, data, state):
        """
        Process a single price bar

        Parameters:
        -----------
        row : pd.Series
            Current price bar data
        current_time_ny : datetime
            Current bar time in NY timezone
        bar_index : int
            Index of current bar
        data : dict
            Dictionary of dataframes for different timeframes
        state : dict
            Current backtest state
        """
        # Get instrument configuration
        config = state['config']

        # Check if we need to handle an open position first
        # Check if we need to handle an open position first
        if state['current_position']:
            self._manage_position(row, current_time_ny, state)

            # Only check for auto-close if the position wasn't already closed by _manage_position
            if state['current_position']:  # Add this check
                # Add end-of-day auto-closure based on AUTO_CLOSE_MINUTES_BEFORE_CLOSE
                from silver_bullet_bot.config import NY_SESSION_END, AUTO_CLOSE_MINUTES_BEFORE_CLOSE

                # Calculate auto-close time
                auto_close_hour = NY_SESSION_END.hour
                auto_close_minute = NY_SESSION_END.minute - AUTO_CLOSE_MINUTES_BEFORE_CLOSE

                # Adjust for negative minutes
                if auto_close_minute < 0:
                    auto_close_hour -= 1
                    auto_close_minute += 60

                # Check for auto-close time
                if current_time_ny.hour == auto_close_hour and current_time_ny.minute == auto_close_minute:
                    self.logger.info(
                        f"Auto-close {AUTO_CLOSE_MINUTES_BEFORE_CLOSE} minutes before market close - Closing any open positions")
                    self._close_position(row, state, row['close'], 'auto_close_before_market_end')

        # Check for trading window
        in_trading_window = self._is_in_trading_window(current_time_ny, config)
        if not in_trading_window:
            return

        # Skip if already have a position or already traded today
        # if state['current_position'] or self._has_traded_today(current_time_ny.date(), state):
        #     return

        # Step 1: Determine HTF bias if not already done
        if state['htf_bias'] is None:
            state['htf_bias'] = self._determine_htf_bias(data, bar_index, state)
            state['potential_setup']['htf_bias'] = state['htf_bias']

            # Calculate Fibonacci levels on H4 data
            if "H4" in data and not data["H4"].empty:
                h4_data = self._get_past_bars("H4", bar_index, 20, data)

                if h4_data is not None and len(h4_data) > 5:
                    # Find recent swing high and low
                    high = h4_data['high'].max()
                    low = h4_data['low'].min()

                    # Calculate Fibonacci levels
                    state['fib_levels'] = calculate_fibonacci_levels(high, low)

            self.logger.info(f"HTF Bias determined: {state['htf_bias']}")

        # Only proceed if we have a clear bias
        if state['htf_bias'] not in ['bullish', 'bearish']:
            return

        # Step 2: Check for liquidity sweep
        if not state['potential_setup']['liquidity_sweep']:
            m1_data = self._get_past_bars("M1", bar_index, 30, data)
            direction = 'buy' if state['htf_bias'] == 'bullish' else 'sell'

            # Detect liquidity sweep
            sweep_data = detect_liquidity_sweep(m1_data, direction)

            if sweep_data:
                state['potential_setup']['liquidity_sweep'] = {
                    'time': sweep_data[0],
                    'price': sweep_data[1],
                    'displacement_index': sweep_data[2]
                }
                self.logger.info(f"Liquidity sweep detected at {sweep_data[0]}, price: {sweep_data[1]:.5f}")

        # Step 3: Look for entry setup
        if state['potential_setup']['liquidity_sweep'] and not state['potential_setup']['entry_pending']:
            direction = 'buy' if state['htf_bias'] == 'bullish' else 'sell'
            m1_data = self._get_past_bars("M1", bar_index, 15, data)

            # First, look for FVG
            fvg_min_size = config.get('fvg_min_size', 5)
            fvg = find_fvg(m1_data, direction, fvg_min_size)

            if fvg:
                # Check if FVG is in correct Fibonacci zone
                if state['fib_levels'] and is_in_fibonacci_zone(fvg['midpoint'], state['fib_levels'], direction):
                    state['potential_setup']['fvg'] = fvg
                    state['potential_setup']['entry_price'] = fvg['midpoint']
                    state['potential_setup']['entry_pending'] = True

                    # Calculate stop loss
                    sweep_price = state['potential_setup']['liquidity_sweep']['price']
                    buffer = config.get('buffer_points', 2) * 0.0001  # Convert to price

                    if direction == 'buy':
                        state['potential_setup']['stop_price'] = sweep_price - buffer
                    else:
                        state['potential_setup']['stop_price'] = sweep_price + buffer

                    self.logger.info(
                        f"FVG entry setup found - Entry: {fvg['midpoint']:.5f}, Stop: {state['potential_setup']['stop_price']:.5f}")
            else:
                # If no FVG, look for Breaker/Order Block
                displacement_index = state['potential_setup']['liquidity_sweep']['displacement_index']
                breaker = find_breaker_or_ob(m1_data, direction, displacement_index)

                if breaker:
                    state['potential_setup']['breaker'] = breaker
                    state['potential_setup']['entry_price'] = breaker['retest_level']
                    state['potential_setup']['entry_pending'] = True

                    # Calculate stop loss
                    sweep_price = state['potential_setup']['liquidity_sweep']['price']
                    buffer = config.get('buffer_points', 2) * 0.0001  # Convert to price

                    if direction == 'buy':
                        state['potential_setup']['stop_price'] = sweep_price - buffer
                    else:
                        state['potential_setup']['stop_price'] = sweep_price + buffer

                    self.logger.info(
                        f"Breaker/OB entry setup found - Entry: {breaker['retest_level']:.5f}, Stop: {state['potential_setup']['stop_price']:.5f}")

        # Step 4: Check for entry conditions
        if state['potential_setup']['entry_pending']:
            direction = 'buy' if state['htf_bias'] == 'bullish' else 'sell'
            entry_price = state['potential_setup']['entry_price']

            # For buy entry, check if price dropped to entry level
            if direction == 'buy' and row['low'] <= entry_price <= row['high']:
                # Entry triggered
                self._execute_entry(row, state, entry_price, direction)
                # ADD THIS LINE - Reset entry_pending to prevent multiple entries
                state['potential_setup']['entry_pending'] = False

            # For sell entry, check if price rose to entry level
            elif direction == 'sell' and row['low'] <= entry_price <= row['high']:
                # Entry triggered
                self._execute_entry(row, state, entry_price, direction)
                # ADD THIS LINE - Reset entry_pending to prevent multiple entries
                state['potential_setup']['entry_pending'] = False

    def _get_past_bars(self, timeframe, current_index, count, data):
        """Get past bars from data for a specific timeframe"""
        if timeframe not in data or data[timeframe].empty:
            return None

        # For M1 (primary timeframe), we can use direct indexing
        if timeframe == "M1":
            start_idx = max(current_index - count, 0)
            return data[timeframe].iloc[start_idx:current_index + 1].copy()

        # For other timeframes, need to match by timestamp
        current_time = data["M1"].iloc[current_index]['time']

        # Filter data frames that are at or before current time
        tf_data = data[timeframe]
        past_data = tf_data[tf_data['time'] <= current_time].tail(count).copy()

        return past_data if not past_data.empty else None

    def _determine_htf_bias(self, data, current_index, state):
        """Determine higher timeframe bias"""
        # Use the current M1 bar's time as reference point
        current_time = data["M1"].iloc[current_index]['time']

        # Get data for each timeframe
        bias_by_tf = {}
        timeframes_to_check = ['DAILY', 'H4', 'H1']

        for tf in timeframes_to_check:
            if tf not in data or data[tf].empty:
                continue

            # Get bars up to current time
            tf_data = data[tf][data[tf]['time'] <= current_time].tail(10).copy()

            if len(tf_data) < 5:
                continue

            # Check if making higher highs and higher lows (bullish)
            higher_highs = tf_data['high'].iloc[-1] > tf_data['high'].iloc[-2] > tf_data['high'].iloc[-3]
            higher_lows = tf_data['low'].iloc[-1] > tf_data['low'].iloc[-2] > tf_data['low'].iloc[-3]

            # Check if making lower highs and lower lows (bearish)
            lower_highs = tf_data['high'].iloc[-1] < tf_data['high'].iloc[-2] < tf_data['high'].iloc[-3]
            lower_lows = tf_data['low'].iloc[-1] < tf_data['low'].iloc[-2] < tf_data['low'].iloc[-3]

            # Assign bias for this timeframe
            if higher_highs and higher_lows:
                bias_by_tf[tf] = 'bullish'
            elif lower_highs and lower_lows:
                bias_by_tf[tf] = 'bearish'
            else:
                bias_by_tf[tf] = 'neutral'

        # Calculate overall bias
        if not bias_by_tf:
            return 'neutral'

        # Count biases
        bullish_count = sum(1 for bias in bias_by_tf.values() if bias == 'bullish')
        bearish_count = sum(1 for bias in bias_by_tf.values() if bias == 'bearish')

        # Determine final bias
        if bullish_count >= 2:
            return 'bullish'
        elif bearish_count >= 2:
            return 'bearish'
        else:
            return 'neutral'

    def _is_in_trading_window(self, current_time_ny, config):
        """Check if current time is in trading window for this instrument"""
        # Get instrument name for logging
        instrument_name = config.get('symbol', 'Unknown')

        # Check if weekend trading is allowed
        if not config.get('trades_on_weekend', False) and current_time_ny.weekday() >= 5:
            if current_time_ny.hour == 0 and current_time_ny.minute == 0:
                self.logger.info(f"{instrument_name}: Weekend - no trading")
            return False

        # Check if within NY trading hours - add more detailed info

        # Log window settings at the start of each hour
        if current_time_ny.minute == 0 and current_time_ny.second < 5:
            self.logger.info(f"Window check for {instrument_name} at {current_time_ny.strftime('%H:%M:%S')} NY")

            # Show the actual window configuration
            if 'windows' in config:
                self.logger.info(f"{instrument_name} has multiple windows defined:")
                for i, window in enumerate(config['windows']):
                    self.logger.info(
                        f"  Window {i + 1}: {window['start'].strftime('%H:%M')} - {window['end'].strftime('%H:%M')} NY")
            else:
                window_start = config.get('window_start')
                window_end = config.get('window_end')
                if window_start and window_end:
                    self.logger.info(
                        f"{instrument_name} window: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')} NY")
                else:
                    self.logger.info(f"{instrument_name} has NO defined window times - check configuration")

        # Check if within instrument-specific window(s)
        is_in_window = False

        if 'windows' in config:
            # Multiple windows case
            for window in config['windows']:
                window_start = window['start']
                window_end = window['end']

                if window_start <= current_time_ny.time() <= window_end:
                    # Log when we're IN a window (every 15 minutes)
                    if current_time_ny.minute % 15 == 0 and current_time_ny.second < 5:
                        self.logger.info(
                            f"{instrument_name} IN TRADING WINDOW at {current_time_ny.strftime('%H:%M:%S')} NY")
                    is_in_window = True
                    break
        else:
            # Single window case
            window_start = config.get('window_start')
            window_end = config.get('window_end')

            if window_start and window_end:
                if window_start <= current_time_ny.time() <= window_end:
                    # Log when we're IN a window (every 15 minutes)
                    if current_time_ny.minute % 15 == 0 and current_time_ny.second < 5:
                        self.logger.info(
                            f"{instrument_name} IN TRADING WINDOW at {current_time_ny.strftime('%H:%M:%S')} NY")
                    is_in_window = True
            else:
                # Use default Silver Bullet window
                is_in_window = is_silver_bullet_window(current_time_ny)
                if is_in_window and current_time_ny.minute % 15 == 0 and current_time_ny.second < 5:
                    self.logger.info(
                        f"{instrument_name} IN DEFAULT SILVER BULLET WINDOW at {current_time_ny.strftime('%H:%M:%S')} NY")

        # if is_in_window:
        #     # Only log once per minute to avoid spam
        #     if current_time_ny.second == 0:
        #         self.logger.debug(
        #             f"{instrument_name} at {current_time_ny.strftime('%Y-%m-%d %H:%M:%S')} NY IS in trading window")
        # elif current_time_ny.hour == 15 and current_time_ny.minute == 30:
        #     # Special log for auto-close time
        #     self.logger.info(
        #         f"{instrument_name} at {current_time_ny.strftime('%Y-%m-%d %H:%M:%S')} NY - Auto-close time check")


        return is_in_window

    def _has_traded_today(self, current_date, state):
        date_str = str(current_date)
        if date_str in state['trades_by_day']:
            # Check instrument max trades per day config
            max_trades = state['config'].get('max_trades_per_day', 1)
            return len(state['trades_by_day'][date_str]) >= max_trades
        return False

    # In backtest_engine.py - around line 600-650 in the _execute_entry method

    def _execute_entry(self, row, state, entry_price, direction):
        """Execute a trade entry"""
        # Calculate lot size
        stop_price = state['potential_setup']['stop_price']

        # Calculate risk amount
        risk_amount = state['balance'] * (MAX_RISK_PERCENT / 100)

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)

        # Get point value for the instrument
        point_value = state['config'].get('point_value', 0.01)

        # Calculate value of stop distance per standard lot
        stop_value_per_lot = stop_distance * point_value

        # Add detailed logging to diagnose the issue
        self.logger.info(f"Position sizing - Balance: ${state['balance']:.2f}, Risk %: {MAX_RISK_PERCENT}")
        self.logger.info(f"Risk amount: ${risk_amount:.2f} (maximum loss per trade)")
        self.logger.info(f"Entry price: {entry_price:.5f}, Stop price: {stop_price:.5f}")
        self.logger.info(f"Stop distance: {stop_distance:.5f}, Point value: {point_value}")
        self.logger.info(f"Stop value per lot: {stop_value_per_lot:.2f}")

        # Calculate required lot size
        if stop_value_per_lot <= 0:
            self.logger.warning("Stop value per lot is zero or negative, using default lot size")
            lot_size = state['config'].get('default_lot_size', 0.1)
        else:
            lot_size = risk_amount / stop_value_per_lot

        # Ensure lot size doesn't exceed maximum
        max_lot_size = min(
            state['config'].get('max_lot_size', 1.0),
            state['config'].get('max_broker_lot_size', float('inf'))
        )

        original_lot_size = lot_size
        lot_size = min(lot_size, max_lot_size)

        # Check if we're hitting the max lot size cap
        if lot_size < original_lot_size:
            self.logger.warning(f"Lot size capped by maximum: {lot_size} (calculated: {original_lot_size:.2f})")

        # Ensure lot size meets minimum (if applicable)
        min_lot_size = state['config'].get('min_lot_size', 0.01)
        if lot_size < min_lot_size:
            self.logger.warning(f"Lot size increased to minimum: {min_lot_size} (calculated: {lot_size:.2f})")
            lot_size = min_lot_size

        self.logger.info(f"Final lot size: {lot_size:.2f}")

        # Rest of the method remains the same...

        # Calculate take profit (if using fixed RR)
        risk_reward_ratio = 2.0  # Default 1:2 risk-reward
        tp_distance = stop_distance * risk_reward_ratio

        take_profit = None
        if direction == 'buy':
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        # Create trade object
        trade = {
            'entry_time': row['time'],
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': stop_price,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'setup_type': 'fvg' if state['potential_setup']['fvg'] else 'breaker',
            'status': 'open',
            'moved_to_breakeven': False,
            'exit_time': None,
            'exit_price': None,
            'exit_reason': None,
            'pnl': 0,
            'pnl_pct': 0,
            'r_multiple': 0
        }

        # Update state
        state['current_position'] = trade

        # Add to trades list and daily trades
        state['trades'].append(trade)
        date_str = str(row['time'].date())
        state['trades_by_day'][date_str].append(trade)
        state['date_stats'][date_str]['trades'] += 1

        self.logger.info(
            f"Trade entry - {direction.upper()} at {entry_price:.5f}, SL: {stop_price:.5f}, Lot size: {lot_size:.2f}")

        # Reset potential setup
        state['potential_setup']['entry_pending'] = False

    def _manage_position(self, row, current_time_ny, state):
        """Manage open position"""
        trade = state['current_position']

        if not trade:
            return

        direction = trade['direction']
        current_price = row['close']  # Use close for evaluation
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']

        # Check if stopped out
        if (direction == 'buy' and row['low'] <= stop_loss) or (direction == 'sell' and row['high'] >= stop_loss):
            self._close_position(row, state, stop_loss, 'stop_loss')
            return

        # Check if take profit hit
        if trade['take_profit']:
            if (direction == 'buy' and row['high'] >= trade['take_profit']) or \
                    (direction == 'sell' and row['low'] <= trade['take_profit']):
                self._close_position(row, state, trade['take_profit'], 'take_profit')
                return

        # Check if need to move to breakeven
        if not trade['moved_to_breakeven']:
            # Calculate 1R distance
            r_distance = abs(entry_price - stop_loss)

            # For buy trade, check if moved up by 1R
            if direction == 'buy' and row['high'] >= entry_price + r_distance:
                trade['stop_loss'] = entry_price
                trade['moved_to_breakeven'] = True
                self.logger.info(f"Moved stop to breakeven at {entry_price:.5f}")

            # For sell trade, check if moved down by 1R
            elif direction == 'sell' and row['low'] <= entry_price - r_distance:
                trade['stop_loss'] = entry_price
                trade['moved_to_breakeven'] = True
                self.logger.info(f"Moved stop to breakeven at {entry_price:.5f}")

        # Check for auto-close based on configuration
        from silver_bullet_bot.config import NY_SESSION_END, AUTO_CLOSE_MINUTES_BEFORE_CLOSE

        # Calculate auto-close time
        auto_close_hour = NY_SESSION_END.hour
        auto_close_minute = NY_SESSION_END.minute - AUTO_CLOSE_MINUTES_BEFORE_CLOSE

        # Adjust for negative minutes
        if auto_close_minute < 0:
            auto_close_hour -= 1
            auto_close_minute += 60

        from datetime import time as dt_time  # This line appears in the code
        # Create the auto-close time
        auto_close_time = dt_time(auto_close_hour, auto_close_minute)

        # Check if we've reached auto-close time
        if current_time_ny.time() >= auto_close_time:
            self.logger.info(f"Auto-close at {auto_close_hour}:{auto_close_minute} NY time - Closing position")
            self._close_position(row, state, current_price, 'auto_close_end_of_day')
            return

    def _close_position(self, row, state, exit_price, exit_reason):
        """Close an open position"""
        if not state['current_position']:
            return

        trade = state['current_position']

        # Set exit details
        trade['exit_time'] = row['time']
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['status'] = 'closed'

        # Calculate profit/loss
        direction = trade['direction']
        entry_price = trade['entry_price']

        if direction == 'buy':
            profit_points = exit_price - entry_price
        else:
            profit_points = entry_price - exit_price

        # Calculate R multiple
        r_distance = abs(entry_price - trade['stop_loss'])
        r_multiple = profit_points / r_distance if r_distance > 0 else 0

        # Calculate monetary profit/loss
        point_value = state['config'].get('point_value', 0.01)
        pnl = profit_points * point_value * trade['lot_size']
        pnl_pct = (pnl / state['balance']) * 100

        # Update trade object
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['r_multiple'] = r_multiple

        # Update account balance and equity
        state['balance'] += pnl
        state['equity'] = state['balance']

        # Update date stats
        date_str = str(row['time'].date())
        if date_str in state['date_stats']:
            if pnl > 0:
                state['date_stats'][date_str]['wins'] += 1
            else:
                state['date_stats'][date_str]['losses'] += 1

            state['date_stats'][date_str]['profit'] += pnl

        # Clear current position
        state['current_position'] = None

        self.logger.info(f"Trade exit - {exit_reason} at {exit_price:.5f}, P/L: {pnl:.2f} ({r_multiple:.2f}R)")

    def _calculate_performance_metrics(self, state):
        """Calculate performance metrics from backtest results"""
        trades = state['trades']

        if not trades:
            self.logger.warning("No trades to calculate metrics")
            return

        # Basic metrics
        state['total_trades'] = len(trades)
        state['closed_trades'] = len([t for t in trades if t['status'] == 'closed'])

        # Profit metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        state['winning_trades'] = len(winning_trades)
        state['losing_trades'] = len(losing_trades)

        state['win_rate'] = state['winning_trades'] / state['closed_trades'] if state['closed_trades'] > 0 else 0

        state['gross_profit'] = sum(t['pnl'] for t in winning_trades)
        state['gross_loss'] = sum(t['pnl'] for t in losing_trades)
        state['net_profit'] = state['gross_profit'] + state['gross_loss']

        state['profit_factor'] = abs(state['gross_profit'] / state['gross_loss']) if state[
                                                                                         'gross_loss'] != 0 else float(
            'inf')

        # Average metrics
        state['avg_profit'] = state['gross_profit'] / len(winning_trades) if winning_trades else 0
        state['avg_loss'] = state['gross_loss'] / len(losing_trades) if losing_trades else 0
        state['avg_r_multiple'] = sum(t['r_multiple'] for t in trades) / len(trades) if trades else 0

        # Drawdown calculation
        balance_history = state['equity_curve']

        # Maximum drawdown calculation
        peak = balance_history[0]
        max_dd = 0
        max_dd_pct = 0

        for balance in balance_history:
            if balance > peak:
                peak = balance
            else:
                dd = peak - balance
                dd_pct = (dd / peak) * 100

                if dd_pct > max_dd_pct:
                    max_dd = dd
                    max_dd_pct = dd_pct

        state['max_drawdown'] = max_dd
        state['max_drawdown_pct'] = max_dd_pct

        # Return statistics
        state['initial_balance'] = self.initial_balance
        state['final_balance'] = state['balance']
        state['absolute_return'] = state['final_balance'] - state['initial_balance']
        state['percent_return'] = (state['absolute_return'] / state['initial_balance']) * 100

        self.logger.info(f"Performance Summary:")
        self.logger.info(f"Total Trades: {state['total_trades']}")
        self.logger.info(f"Win Rate: {state['win_rate']:.2%}")
        self.logger.info(f"Net Profit: ${state['net_profit']:.2f} ({state['percent_return']:.2f}%)")
        self.logger.info(f"Profit Factor: {state['profit_factor']:.2f}")
        self.logger.info(f"Average R: {state['avg_r_multiple']:.2f}")
        self.logger.info(f"Maximum Drawdown: ${state['max_drawdown']:.2f} ({state['max_drawdown_pct']:.2f}%)")

    def save_results(self, results, file_format='json'):
        """Save backtest results to file"""
        if not results:
            self.logger.error("No results to save")
            return None

        instrument = results['instrument']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if file_format == 'json':
            # Create a copy of results to clean up for JSON serialization
            json_results = {}
            for key, value in results.items():
                # Skip non-serializable objects or large data structures
                if key in ['data_loader', 'logger', 'current_position']:
                    continue

                # Handle datetime objects
                if key == 'trades':
                    # Clone and convert datetime objects in trades
                    json_results[key] = []
                    for trade in value:
                        trade_copy = dict(trade)
                        for time_field in ['entry_time', 'exit_time']:
                            if time_field in trade_copy and trade_copy[time_field] is not None:
                                trade_copy[time_field] = trade_copy[time_field].isoformat()
                        json_results[key].append(trade_copy)
                else:
                    json_results[key] = value

            # Save to file
            filename = os.path.join(self.output_dir, f"backtest_{instrument}_{timestamp}.json")
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)

            self.logger.info(f"Results saved to {filename}")
            return filename

        elif file_format == 'csv':
            # Save trades to CSV
            if 'trades' in results and results['trades']:
                # Clone and prepare trades for CSV
                trades_for_csv = []
                for trade in results['trades']:
                    trade_copy = dict(trade)
                    # Convert datetime objects
                    for time_field in ['entry_time', 'exit_time']:
                        if time_field in trade_copy and trade_copy[time_field] is not None:
                            trade_copy[time_field] = trade_copy[time_field].isoformat()
                    trades_for_csv.append(trade_copy)

                # Create DataFrame and save
                trades_df = pd.DataFrame(trades_for_csv)
                filename = os.path.join(self.output_dir, f"backtest_{instrument}_trades_{timestamp}.csv")
                trades_df.to_csv(filename, index=False)

                self.logger.info(f"Trades saved to {filename}")
                return filename

        return None
