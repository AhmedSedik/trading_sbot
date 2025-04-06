# silver_bullet_bot/core/utils.py

import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import math

from silver_bullet_bot.config import (
    MAX_RISK_PERCENT, TIMEFRAMES,
    LIQUIDITY_SWEEP_MIN_PIPS, FVG_MIN_SIZE_POINTS,
    SAFE_BUFFER_POINTS
)


def setup_utils():
    """Set up utilities and return logger"""
    logger = logging.getLogger('silver_bullet')
    logger.info("Initializing strategy utilities")
    return logger


def copy_rates_from_pos(symbol, timeframe, start_pos, count):
    """
    Get historical bars from MT5, handling errors and conversions

    Parameters:
    -----------
    symbol : str
        The broker-specific symbol
    timeframe : str
        Timeframe name as in TIMEFRAMES config
    start_pos : int
        Starting position from the current bar (0 means current bar)
    count : int
        Number of bars to retrieve

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with OHLC data, or None if error
    """
    logger = logging.getLogger('silver_bullet')

    # Get timeframe constant value from MT5
    tf = getattr(mt5, TIMEFRAMES[timeframe])

    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)

    if rates is None:
        logger.error(f"Failed to get rates for {symbol} {timeframe}. Error: {mt5.last_error()}")
        return None

    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)

    # Convert time to datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')

    return df


def detect_liquidity_sweep(df, direction):
    """
    Detect a liquidity sweep (stop hunt) in recent price action

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with recent price data
    direction : str
        'buy' to look for lows being swept, 'sell' to look for highs being swept

    Returns:
    --------
    tuple or None
        (sweep_time, sweep_price, displacement_index) if sweep detected, None otherwise
    """
    logger = logging.getLogger('silver_bullet')

    if df is None or len(df) < 5:
        logger.debug("Not enough data to detect liquidity sweep")
        return None

    # Log relevant price data for analysis
    logger.info(f"Analyzing for {direction} liquidity sweep - Recent price action:")
    for i in range(min(5, len(df))):
        candle = df.iloc[i]
        logger.info(
            f"Candle {i} at {candle['time']} - O: {candle['open']:.5f}, H: {candle['high']:.5f}, L: {candle['low']:.5f}, C: {candle['close']:.5f}")

    # For buy setup, look for low being swept then price recovering
    if direction.lower() == 'buy':
        # Find the lowest low in prior candles (skip most recent which could be currently forming)
        prior_low_idx = df['low'].iloc[1:].idxmin()
        prior_low = df.loc[prior_low_idx, 'low']
        prior_low_time = df.loc[prior_low_idx, 'time']

        logger.info(f"Buy setup - Looking for sweep of prior low: {prior_low:.5f} from {prior_low_time}")

        # Check if recent candle(s) broke below this low
        for i in range(min(3, len(df))):  # Check most recent 3 candles
            sweep_condition = df['low'].iloc[i] < prior_low
            logger.info(
                f"Candle {i} sweep check - Low: {df['low'].iloc[i]:.5f} vs Prior low: {prior_low:.5f} - Swept: {sweep_condition}")

            if sweep_condition:
                # Calculate sweep magnitude
                sweep_magnitude = abs(df['low'].iloc[i] - prior_low)
                min_sweep = LIQUIDITY_SWEEP_MIN_PIPS * 0.0001  # Convert pips to price

                logger.info(
                    f"Potential buy sweep detected - Candle low: {df['low'].iloc[i]:.5f}, Prior low: {prior_low:.5f}")
                logger.info(f"Sweep magnitude: {sweep_magnitude:.5f} pips, Minimum required: {min_sweep:.5f} pips")

                # We've found a potential sweep - now check for recovery/displacement
                for j in range(i + 1, min(i + 4, len(df))):
                    # If price recovered above the prior low with a bullish candle
                    recovery_condition = df['close'].iloc[j] > prior_low and df['close'].iloc[j] > df['open'].iloc[j]
                    logger.info(
                        f"Displacement check - Candle {j} - Close: {df['close'].iloc[j]:.5f} vs Prior low: {prior_low:.5f}, Bullish: {df['close'].iloc[j] > df['open'].iloc[j]} - Recovery: {recovery_condition}")

                    if recovery_condition:
                        # Check if sweep magnitude meets minimum criteria
                        if sweep_magnitude >= min_sweep:
                            logger.info(
                                f"Valid buy-side liquidity sweep at {df['time'].iloc[i]}, swept low: {prior_low:.5f}, sweep magnitude: {sweep_magnitude:.5f}, recovery at candle {j}")
                            return df['time'].iloc[i], df['low'].iloc[i], j
                        else:
                            logger.info(
                                f"Buy sweep magnitude too small: {sweep_magnitude:.5f} < {min_sweep:.5f} - Ignoring")

    # For sell setup, look for high being swept then price dropping
    elif direction.lower() == 'sell':
        # Find the highest high in prior candles
        prior_high_idx = df['high'].iloc[1:].idxmax()
        prior_high = df.loc[prior_high_idx, 'high']
        prior_high_time = df.loc[prior_high_idx, 'time']

        logger.info(f"Sell setup - Looking for sweep of prior high: {prior_high:.5f} from {prior_high_time}")

        # Check if recent candle(s) broke above this high
        for i in range(min(3, len(df))):  # Check most recent 3 candles
            sweep_condition = df['high'].iloc[i] > prior_high
            logger.info(
                f"Candle {i} sweep check - High: {df['high'].iloc[i]:.5f} vs Prior high: {prior_high:.5f} - Swept: {sweep_condition}")

            if sweep_condition:
                # Calculate sweep magnitude
                sweep_magnitude = abs(df['high'].iloc[i] - prior_high)
                min_sweep = LIQUIDITY_SWEEP_MIN_PIPS * 0.0001  # Convert pips to price

                logger.info(
                    f"Potential sell sweep detected - Candle high: {df['high'].iloc[i]:.5f}, Prior high: {prior_high:.5f}")
                logger.info(f"Sweep magnitude: {sweep_magnitude:.5f} pips, Minimum required: {min_sweep:.5f} pips")

                # We've found a potential sweep - now check for recovery/displacement
                for j in range(i + 1, min(i + 4, len(df))):
                    # If price dropped below the prior high with a bearish candle
                    recovery_condition = df['close'].iloc[j] < prior_high and df['close'].iloc[j] < df['open'].iloc[j]
                    logger.info(
                        f"Displacement check - Candle {j} - Close: {df['close'].iloc[j]:.5f} vs Prior high: {prior_high:.5f}, Bearish: {df['close'].iloc[j] < df['open'].iloc[j]} - Recovery: {recovery_condition}")

                    if recovery_condition:
                        # Check if sweep magnitude meets minimum criteria
                        if sweep_magnitude >= min_sweep:
                            logger.info(
                                f"Valid sell-side liquidity sweep at {df['time'].iloc[i]}, swept high: {prior_high:.5f}, sweep magnitude: {sweep_magnitude:.5f}, recovery at candle {j}")
                            return df['time'].iloc[i], df['high'].iloc[i], j
                        else:
                            logger.info(
                                f"Sell sweep magnitude too small: {sweep_magnitude:.5f} < {min_sweep:.5f} - Ignoring")

    logger.info(f"No valid {direction} liquidity sweep detected after analysis")
    return None


def find_fvg(df, direction, min_size=None):
    """
    Find a Fair Value Gap (FVG) in recent price action

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with recent price data
    direction : str
        'buy' for bullish FVG, 'sell' for bearish FVG
    min_size : float, optional
        Minimum size of FVG in points, defaults to FVG_MIN_SIZE_POINTS from config

    Returns:
    --------
    dict or None
        FVG details if found, None otherwise
    """
    logger = logging.getLogger('silver_bullet')

    if df is None or len(df) < 3:
        logger.debug("Not enough data to find FVG")
        return None

    if min_size is None:
        min_size = FVG_MIN_SIZE_POINTS

    # Log candles for analysis
    logger.info(f"Analyzing for {direction} FVG with minimum size {min_size} - Recent candles:")
    for i in range(min(5, len(df))):
        candle = df.iloc[i]
        logger.info(
            f"Candle {i} at {candle['time']} - O: {candle['open']:.5f}, H: {candle['high']:.5f}, L: {candle['low']:.5f}, C: {candle['close']:.5f}")

    min_size_price = min_size * 0.0001  # Convert to price
    logger.info(f"Minimum FVG size in price: {min_size_price:.5f}")

    # For a bullish FVG (upside imbalance)
    if direction.lower() == 'buy':
        logger.info(f"Looking for bullish FVG - Need candle 1 low > candle 3 high")
        # Look for a gap where candle 1's low is higher than candle 3's high
        # (in a series of 3 consecutive candles forming an imbalance)
        for i in range(len(df) - 2):
            candle1 = df.iloc[i]
            candle2 = df.iloc[i + 1]
            candle3 = df.iloc[i + 2]

            # Check if there's a gap between the 1st and 3rd candle
            gap_condition = candle1['low'] > candle3['high']
            logger.info(
                f"FVG check at candles {i}-{i + 2}: Candle 1 Low ({candle1['low']:.5f}) > Candle 3 High ({candle3['high']:.5f}) = {gap_condition}")

            if gap_condition:
                # Calculate gap size
                gap_size = candle1['low'] - candle3['high']

                logger.info(f"Potential bullish FVG at {candle2['time']} - Gap size: {gap_size:.5f}")
                logger.info(f"FVG range: {candle3['high']:.5f} - {candle1['low']:.5f}")

                # Check if the gap meets minimum size requirement
                size_condition = gap_size >= min_size_price
                logger.info(f"Size check: {gap_size:.5f} >= {min_size_price:.5f} = {size_condition}")

                if size_condition:
                    # Calculate midpoint
                    midpoint = candle3['high'] + (gap_size / 2)

                    fvg = {
                        'type': 'bullish',
                        'time': candle2['time'],
                        'high': candle1['low'],
                        'low': candle3['high'],
                        'midpoint': midpoint,
                        'size': gap_size
                    }

                    logger.info(f"Valid bullish FVG found at {fvg['time']}")
                    logger.info(
                        f"FVG details - Range: {fvg['low']:.5f} - {fvg['high']:.5f}, Midpoint: {fvg['midpoint']:.5f}, Size: {fvg['size']:.5f}")
                    return fvg
                else:
                    logger.info(f"Bullish FVG too small, rejecting")

    # For a bearish FVG (downside imbalance)
    elif direction.lower() == 'sell':
        logger.info(f"Looking for bearish FVG - Need candle 1 high < candle 3 low")
        # Look for a gap where candle 1's high is lower than candle 3's low
        for i in range(len(df) - 2):
            candle1 = df.iloc[i]
            candle2 = df.iloc[i + 1]
            candle3 = df.iloc[i + 2]

            # Check if there's a gap between the 1st and 3rd candle
            gap_condition = candle1['high'] < candle3['low']
            logger.info(
                f"FVG check at candles {i}-{i + 2}: Candle 1 High ({candle1['high']:.5f}) < Candle 3 Low ({candle3['low']:.5f}) = {gap_condition}")

            if gap_condition:
                # Calculate gap size
                gap_size = candle3['low'] - candle1['high']

                logger.info(f"Potential bearish FVG at {candle2['time']} - Gap size: {gap_size:.5f}")
                logger.info(f"FVG range: {candle1['high']:.5f} - {candle3['low']:.5f}")

                # Check if the gap meets minimum size requirement
                size_condition = gap_size >= min_size_price
                logger.info(f"Size check: {gap_size:.5f} >= {min_size_price:.5f} = {size_condition}")

                if size_condition:
                    # Calculate midpoint
                    midpoint = candle1['high'] + (gap_size / 2)

                    fvg = {
                        'type': 'bearish',
                        'time': candle2['time'],
                        'high': candle3['low'],
                        'low': candle1['high'],
                        'midpoint': midpoint,
                        'size': gap_size
                    }

                    logger.info(f"Valid bearish FVG found at {fvg['time']}")
                    logger.info(
                        f"FVG details - Range: {fvg['low']:.5f} - {fvg['high']:.5f}, Midpoint: {fvg['midpoint']:.5f}, Size: {fvg['size']:.5f}")
                    return fvg
                else:
                    logger.info(f"Bearish FVG too small, rejecting")

    logger.info(f"No valid {direction} FVG found after analyzing {len(df)} candles")
    return None


def find_breaker_or_ob(df, direction, displacement_index):
    """
    Find a Breaker Block or Order Block if no FVG is found

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with recent price data
    direction : str
        'buy' for bullish setup, 'sell' for bearish setup
    displacement_index : int
        Index of the displacement candle

    Returns:
    --------
    dict or None
        Breaker/OB details if found, None otherwise
    """
    logger = logging.getLogger('silver_bullet')

    if df is None or len(df) < displacement_index + 1:
        logger.debug("Not enough data to find Breaker/OB")
        return None

    logger.info(
        f"Analyzing for {direction} Breaker/Order Block after displacement candle at index {displacement_index}")

    # Log displacement candle
    if displacement_index < len(df):
        disp_candle = df.iloc[displacement_index]
        logger.info(
            f"Displacement candle: O: {disp_candle['open']:.5f}, H: {disp_candle['high']:.5f}, L: {disp_candle['low']:.5f}, C: {disp_candle['close']:.5f}, Time: {disp_candle['time']}")

    # For a bullish setup (looking for bearish OB that becomes bullish breaker)
    if direction.lower() == 'buy':
        logger.info(f"Looking for bullish breaker (last bearish candle before displacement)")
        # Look for the last bearish candle before the displacement
        for i in range(displacement_index + 1, min(displacement_index + 5, len(df))):
            candle = df.iloc[i]
            is_bearish = candle['close'] < candle['open']
            logger.info(
                f"Checking candle {i} - Bearish: {is_bearish}, O: {candle['open']:.5f}, H: {candle['high']:.5f}, L: {candle['low']:.5f}, C: {candle['close']:.5f}")

            if is_bearish:  # Bearish candle
                # This is a potential order block
                midpoint = (candle['high'] + candle['low']) / 2
                retest_level = candle['low'] + (candle['high'] - candle['low']) * 0.5  # 50% level

                ob = {
                    'type': 'bullish_breaker',
                    'time': candle['time'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'midpoint': midpoint,
                    'retest_level': retest_level
                }

                logger.info(f"Found bullish breaker/OB at {ob['time']}")
                logger.info(
                    f"Breaker details - Range: {ob['low']:.5f} - {ob['high']:.5f}, Midpoint: {ob['midpoint']:.5f}, Retest level: {ob['retest_level']:.5f}")
                return ob

    # For a bearish setup (looking for bullish OB that becomes bearish breaker)
    elif direction.lower() == 'sell':
        logger.info(f"Looking for bearish breaker (last bullish candle before displacement)")
        # Look for the last bullish candle before the displacement
        for i in range(displacement_index + 1, min(displacement_index + 5, len(df))):
            candle = df.iloc[i]
            is_bullish = candle['close'] > candle['open']
            logger.info(
                f"Checking candle {i} - Bullish: {is_bullish}, O: {candle['open']:.5f}, H: {candle['high']:.5f}, L: {candle['low']:.5f}, C: {candle['close']:.5f}")

            if is_bullish:  # Bullish candle
                # This is a potential order block
                midpoint = (candle['high'] + candle['low']) / 2
                retest_level = candle['high'] - (candle['high'] - candle['low']) * 0.5  # 50% level

                ob = {
                    'type': 'bearish_breaker',
                    'time': candle['time'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'midpoint': midpoint,
                    'retest_level': retest_level
                }

                logger.info(f"Found bearish breaker/OB at {ob['time']}")
                logger.info(
                    f"Breaker details - Range: {ob['low']:.5f} - {ob['high']:.5f}, Midpoint: {ob['midpoint']:.5f}, Retest level: {ob['retest_level']:.5f}")
                return ob

    logger.info(f"No valid {direction} breaker/OB found")
    return None


def calculate_lot_size(account_balance, stop_price, entry_price, instrument_config, open_positions_lot_sum=0):
    """
    Calculate appropriate lot size based on risk parameters

    Parameters:
    -----------
    account_balance : float
        Current account balance
    stop_price : float
        Stop loss price
    entry_price : float
        Entry price
    instrument_config : dict
        Instrument configuration from config

    Returns:
    --------
    float
        Calculated lot size
    """
    logger = logging.getLogger('silver_bullet')

    # Calculate risk amount in account currency
    # Original lot size calculation
    risk_amount = account_balance * (MAX_RISK_PERCENT / 100)
    stop_distance = abs(entry_price - stop_price)
    point_value = instrument_config.get('point_value', 0.01)
    stop_value_per_lot = stop_distance * point_value

    # Log detailed risk calculations
    logger.info(f"Risk calculation - Account balance: {account_balance:.2f}, Risk %: {MAX_RISK_PERCENT}")
    logger.info(f"Risk amount: {risk_amount:.2f} (maximum loss per trade)")
    logger.info(f"Entry price: {entry_price:.5f}, Stop price: {stop_price:.5f}")
    logger.info(f"Stop distance: {stop_distance:.5f}, Point value: {point_value}")
    logger.info(f"Stop value per lot: {stop_value_per_lot:.2f}")

    # Calculate base lot size
    if stop_value_per_lot <= 0:
        lot_size = instrument_config.get('default_lot_size', 0.1)
    else:
        lot_size = risk_amount / stop_value_per_lot

    # Get max settings
    max_lot_size = min(
        instrument_config.get('max_lot_size', 1.0),
        instrument_config.get('max_broker_lot_size', float('inf'))
    )
    max_concurrent_lots = instrument_config.get('max_concurrent_lots', max_lot_size)

    # Calculate available lot size considering open positions
    available_lot_size = max_concurrent_lots - open_positions_lot_sum

    # Adjust lot size to respect maximum concurrent exposure
    lot_size = min(lot_size, available_lot_size, max_lot_size)

    # Ensure minimum lot size
    min_lot_size = instrument_config.get('min_lot_size', 0.01)
    lot_size = max(lot_size, min_lot_size)

    logger.info(
        f"Adjusted lot size: {lot_size:.2f} (after considering {open_positions_lot_sum} lots in open positions)")

    return lot_size


def determine_bias(symbol, timeframes=None):
    """
    Determine market bias based on multiple timeframes

    Parameters:
    -----------
    symbol : str
        The broker-specific symbol
    timeframes : list, optional
        List of timeframes to analyze, defaults to ['DAILY', 'H4', 'H1']

    Returns:
    --------
    str
        'bullish', 'bearish', or 'neutral'
    """
    logger = logging.getLogger('silver_bullet')

    if timeframes is None:
        timeframes = ['DAILY', 'H4', 'H1']

    bias_scores = []

    for tf in timeframes:
        # Get data for this timeframe
        df = copy_rates_from_pos(symbol, tf, 0, 10)

        if df is None or len(df) < 5:
            logger.warning(f"Not enough data for {tf} bias determination, skipping")
            continue

        # Simple bias determination based on last N candles
        # Check if making higher highs and higher lows (bullish)
        higher_highs = df['high'].iloc[0] > df['high'].iloc[1] > df['high'].iloc[2]
        higher_lows = df['low'].iloc[0] > df['low'].iloc[1] > df['low'].iloc[2]

        # Check if making lower highs and lower lows (bearish)
        lower_highs = df['high'].iloc[0] < df['high'].iloc[1] < df['high'].iloc[2]
        lower_lows = df['low'].iloc[0] < df['low'].iloc[1] < df['low'].iloc[2]

        # Assign a score for this timeframe
        if higher_highs and higher_lows:
            bias_scores.append(1)  # Bullish
        elif lower_highs and lower_lows:
            bias_scores.append(-1)  # Bearish
        else:
            bias_scores.append(0)  # Neutral

    # Calculate overall bias
    if not bias_scores:
        return 'neutral'

    avg_score = sum(bias_scores) / len(bias_scores)

    if avg_score > 0.3:
        return 'bullish'
    elif avg_score < -0.3:
        return 'bearish'
    else:
        return 'neutral'


def calculate_fibonacci_levels(high, low):
    """
    Calculate Fibonacci retracement levels

    Parameters:
    -----------
    high : float
        High price
    low : float
        Low price

    Returns:
    --------
    dict
        Dictionary with Fibonacci levels
    """
    range_size = high - low

    levels = {
        '0.0': low,
        '23.6': low + range_size * 0.236,
        '38.2': low + range_size * 0.382,
        '50.0': low + range_size * 0.5,
        '61.8': low + range_size * 0.618,
        '78.6': low + range_size * 0.786,
        '100.0': high
    }

    return levels


def is_in_fibonacci_zone(price, fib_levels, direction):
    """
    Check if a price level is in the correct Fibonacci zone

    Parameters:
    -----------
    price : float
        Price to check
    fib_levels : dict
        Dictionary with Fibonacci levels
    direction : str
        'buy' or 'sell'

    Returns:
    --------
    bool
        True if price is in the correct zone, False otherwise
    """
    logger = logging.getLogger('silver_bullet')

    # Log the analysis
    logger.info(f"Checking if price {price:.5f} is in correct Fibonacci zone for {direction}")
    logger.info(
        f"Fibonacci levels: 0%: {fib_levels['0.0']:.5f}, 50%: {fib_levels['50.0']:.5f}, 100%: {fib_levels['100.0']:.5f}")

    # For buy, we want price to be below 50% (in discount)
    if direction.lower() == 'buy':
        result = price < fib_levels['50.0']
        logger.info(f"Buy check: Price {price:.5f} < 50% level {fib_levels['50.0']:.5f} = {result}")
        return result

    # For sell, we want price to be above 50% (in premium)
    elif direction.lower() == 'sell':
        result = price > fib_levels['50.0']
        logger.info(f"Sell check: Price {price:.5f} > 50% level {fib_levels['50.0']:.5f} = {result}")
        return result

    return False