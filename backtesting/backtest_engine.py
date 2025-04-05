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
    convert_mt5_to_ny, is_silver_bullet_window, is_ny_trading_time
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
            m1_data['time'] = m1_data['time'].dt.tz_localize('UTC')

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

        # Process each M1 bar
        for i in tqdm(range(len(m1_data)), desc=f"Backtesting {instrument_name}"):
            # Current bar row
            row = m1_data.iloc[i]
            current_time = row['time']
            current_date = current_time.date()

            # Convert to NY time (for session checks)
            current_time_ny = current_time.tz_convert('America/New_York')
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
        if state['current_position']:
            self._manage_position(row, current_time_ny, state)

        # Check for trading window
        in_trading_window = self._is_in_trading_window(current_time_ny, config)
        if not in_trading_window:
            return

        # Skip if already have a position or already traded today
        if state['current_position'] or self._has_traded_today(current_time_ny.date(), state):
            return

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

            # For sell entry, check if price rose to entry level
            elif direction == 'sell' and row['low'] <= entry_price <= row['high']:
                # Entry triggered
                self._execute_entry(row, state, entry_price, direction)

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
        # Check if weekend trading is allowed
        if not config.get('trades_on_weekend', False) and current_time_ny.weekday() >= 5:
            return False

        # Check if within NY trading hours
        if not is_ny_trading_time(current_time_ny):
            return False

        # Check if within instrument-specific window(s)
        if 'windows' in config:
            # Multiple windows case
            for window in config['windows']:
                window_start = window['start']
                window_end = window['end']

                if window_start <= current_time_ny.time() <= window_end:
                    return True
            return False
        else:
            # Single window case
            window_start = config.get('window_start')
            window_end = config.get('window_end')

            if window_start and window_end:
                return window_start <= current_time_ny.time() <= window_end
            else:
                # Use default Silver Bullet window
                return is_silver_bullet_window(current_time_ny)

    def _has_traded_today(self, current_date, state):
        date_str = str(current_date)
        if date_str in state['trades_by_day']:
            # Check instrument max trades per day config
            max_trades = state['config'].get('max_trades_per_day', 1)
            return len(state['trades_by_day'][date_str]) >= max_trades
        return False

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

        # Calculate required lot size
        lot_size = risk_amount / stop_value_per_lot if stop_value_per_lot > 0 else state['config'].get(
            'default_lot_size', 0.1)

        # Ensure lot size doesn't exceed maximum
        max_lot_size = min(
            state['config'].get('max_lot_size', 1.0),
            state['config'].get('max_broker_lot_size', float('inf'))
        )
        lot_size = min(lot_size, max_lot_size)

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

        # Check for time-based exit (10:35 AM NY time)
        from datetime import time as dt_time  # Add this at the top of the file
        # Then in the _manage_position method:
        exit_time = dt_time(10, 35)  # 10:35 AM

        if current_time_ny.time() >= exit_time:
            self._close_position(row, state, current_price, 'time_exit')
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