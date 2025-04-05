# silver_bullet_bot/main.py

import os
import sys
import logging
import time
import argparse
import signal
from datetime import datetime
import MetaTrader5 as mt5

from silver_bullet_bot.config import (
    MT5_BROKERS, DEFAULT_BROKER, ACTIVE_INSTRUMENTS,
    LOG_DIR, LOG_LEVEL, LOG_FORMAT, DEBUG_MODE, MT5_TO_NY_OFFSET
)
from silver_bullet_bot.brokers.symbol_mapper import SymbolMapper
from silver_bullet_bot.core.strategy import SilverBulletStrategy
from silver_bullet_bot.core.timezone_utils import (
    convert_mt5_to_ny, format_time_for_display
)

# Global variables for clean shutdown
running = True
strategies = []


def setup_logging():
    """Set up logging"""
    # Create base directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Set up root logger
    logger = logging.getLogger('silver_bullet')
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Create file handler
    log_file = os.path.join(LOG_DIR, 'general.log')
    file_handler = logging.FileHandler(log_file)

    # Create console handler
    console_handler = logging.StreamHandler()

    # Set format
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def signal_handler(sig, frame):
    """Handle interrupt signals for clean shutdown"""
    global running
    logger = logging.getLogger('silver_bullet')
    logger.info("Received shutdown signal, closing gracefully...")
    running = False


def connect_to_mt5(broker_name=None):
    """
    Connect to MT5 terminal

    Parameters:
    -----------
    broker_name : str, optional
        Broker name from MT5_BROKERS config, defaults to DEFAULT_BROKER

    Returns:
    --------
    bool
        True if connection successful, False otherwise
    """
    logger = logging.getLogger('silver_bullet')

    # If broker_name not provided, use default
    if not broker_name:
        broker_name = DEFAULT_BROKER

    # Check if broker exists in config
    if broker_name not in MT5_BROKERS:
        logger.error(f"Broker '{broker_name}' not found in configuration")
        return False

    broker_config = MT5_BROKERS[broker_name]

    # Initialize MT5 connection
    logger.info(f"Connecting to MT5 terminal for {broker_config['name']}...")

    if not mt5.initialize(broker_config['path']):
        logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
        return False

    # Display MT5 version
    mt5_version = mt5.version()
    if mt5_version:
        logger.info(f"Connected to MetaTrader 5 version {mt5_version[0]}.{mt5_version[1]}")
    else:
        logger.warning("Could not get MetaTrader 5 version")

    # Check if there's a connection to a trading account
    if not mt5.account_info():
        logger.error("No trading account connected. Please log in to your trading account in the MT5 terminal.")
        mt5.shutdown()
        return False

    # Display account info
    account_info = mt5.account_info()._asdict()
    logger.info(f"Connected to account {account_info['login']} ({account_info['server']})")
    logger.info(f"Balance: {account_info['balance']}, Equity: {account_info['equity']}")

    # Display time information
    current_local_time = datetime.now()

    # Get MT5 time using a reliable method
    try:
        current_mt5_time_timestamp = mt5.symbol_info_tick("EURUSD").time
        current_mt5_time = datetime.fromtimestamp(current_mt5_time_timestamp)
        logger.info(f"MT5 Tick Time Timestamp: {current_mt5_time_timestamp}")
    except Exception as e:
        logger.warning(f"Error getting MT5 time: {e}")
        current_mt5_time = current_local_time

    # Calculate NY time from local time instead of MT5 time
    ny_time = convert_mt5_to_ny(current_local_time)

    logger.info("Bot Timestamp Summary:")
    logger.info(f"Local Time: {current_local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"MT5 Server Time: {current_mt5_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"New York Time (calculated from local): {ny_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    return True


def run_bot(broker_name=None):
    """
    Run the trading bot

    Parameters:
    -----------
    broker_name : str, optional
        Broker name from MT5_BROKERS config, defaults to DEFAULT_BROKER
    """
    global running, strategies

    # Set up logging
    logger = setup_logging()

    # Print a startup banner with ASCII-safe characters
    startup_banner = """
    +----------------------------------------------------------+
    |                                                          |
    |         ICT SILVER BULLET TRADING BOT - STARTING         |
    |                                                          |
    +----------------------------------------------------------+    
    """
    logger.info(startup_banner)

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Bot starting on {current_date} at {current_time}")

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Connect to MT5
    if not connect_to_mt5(broker_name):
        logger.error("Failed to connect to MT5, exiting")
        return

    try:
        # Set up symbol mapper
        symbol_mapper = SymbolMapper()

        # Initialize strategies for active instruments
        for instrument in ACTIVE_INSTRUMENTS:
            # Get broker-specific symbol
            symbol = symbol_mapper.get_broker_symbol(instrument)

            if not symbol:
                logger.warning(f"Could not find symbol mapping for {instrument}, skipping")
                continue

            # Create strategy for this instrument
            try:
                strategy = SilverBulletStrategy(instrument, symbol, symbol_mapper)
                strategies.append(strategy)

                # Get current NY time for logging
                ny_time = convert_mt5_to_ny(datetime.now())
                logger.info(
                    f"[{ny_time.strftime('%H:%M:%S')} NY] Initialized strategy for {instrument} (Symbol: {symbol})")
            except Exception as e:
                logger.error(f"Failed to initialize strategy for {instrument}: {e}")

        if not strategies:
            logger.error("No strategies could be initialized, exiting")
            return

        # Main loop
        ny_time = convert_mt5_to_ny(datetime.now())

        # Print a main loop banner with time info (using ASCII-safe characters)
        main_loop_banner = """
        +----------------------------------------------------------+
        |                                                          |
        |               MAIN TRADING LOOP STARTED                  |
        |                                                          |
        +----------------------------------------------------------+    
        """
        logger.info(main_loop_banner)

        # Print detailed time information
        logger.info(f"Current timestamps:")
        current_local_time = datetime.now()
        logger.info(f"Local Time: {current_local_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            mt5_time_timestamp = mt5.symbol_info_tick("EURUSD").time
            mt5_time = datetime.fromtimestamp(mt5_time_timestamp)
            logger.info(f"MT5 Server Time: {mt5_time.strftime('%Y-%m-%d %H:%M:%S')} (timestamp: {mt5_time_timestamp})")
        except Exception as e:
            logger.warning(f"Error getting MT5 time: {e}")
            mt5_time = current_local_time
            logger.info(f"MT5 Server Time: Error retrieving time")

        # Always calculate NY time from local time for consistency
        ny_time = convert_mt5_to_ny(current_local_time)
        logger.info(f"New York Time (from local): {ny_time.strftime('%Y-%m-%d %H:%M:%S')}")

        utc_time = datetime.utcnow()
        logger.info(f"UTC Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)

        # Get time zone offset information
        try:
            # Calculate timezone offsets
            local_tz = datetime.now().astimezone().tzinfo
            utc_offset = datetime.now().astimezone().utcoffset().total_seconds() / 3600

            # Determine NY offset from UTC
            import pytz
            ny_tz = pytz.timezone('America/New_York')
            ny_now = datetime.now(ny_tz)
            ny_offset = ny_now.utcoffset().total_seconds() / 3600

            # Calculate MT5 offset from config
            mt5_offset = MT5_TO_NY_OFFSET - ny_offset

            # Log the timezone information
            logger.info(f"Timezone Information:")
            logger.info(f"Local Timezone: {local_tz}, UTC Offset: {utc_offset:+.1f} hours")
            logger.info(f"MT5 Server Timezone: UTC{mt5_offset:+.1f} hours")
            logger.info(f"New York Timezone: UTC{ny_offset:+.1f} hours")
            logger.info(f"MT5 to NY Offset: {MT5_TO_NY_OFFSET} hours")
            logger.info("=" * 50)
        except Exception as e:
            logger.warning(f"Could not determine timezone offsets: {e}")

        logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] Entering main loop...")

        # Track current date for day changes
        current_day = ny_time.date()

        # Track end-of-session logging
        end_of_session_logged = False

        while running:
            # Get current time
            current_time = datetime.now()

            # Convert to NY time
            ny_time = convert_mt5_to_ny(current_time)

            # Heartbeat log every 10 seconds to confirm bot is running
            if current_time.second % 10 == 0:
                logger.debug(f"[{ny_time.strftime('%H:%M:%S')} NY] Bot heartbeat - running")

            # Display time info in debug mode
            if DEBUG_MODE:
                logger.debug(f"Current time: {format_time_for_display(current_time)}")
                logger.debug(f"NY time: {ny_time.strftime('%H:%M:%S')}")

            # Check for new day (for resetting daily stats)
            if ny_time.date() != current_day:
                logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] New trading day detected")
                current_day = ny_time.date()
                end_of_session_logged = False

                # Get account information for new day
                account_info = mt5.account_info()
                if account_info:
                    logger.info(
                        f"[{ny_time.strftime('%H:%M:%S')} NY] New day starting account balance: {account_info.balance}, equity: {account_info.equity}")

            # Check for end of session (6pm NY time)
            if ny_time.hour == 18 and ny_time.minute == 0 and not end_of_session_logged:
                # Log end of session
                account_info = mt5.account_info()
                if account_info:
                    logger.info(
                        f"[{ny_time.strftime('%H:%M:%S')} NY] End of trading session - Final account balance: {account_info.balance}, equity: {account_info.equity}")
                end_of_session_logged = True

            # Run strategy iteration for each instrument
            active_count = 0
            for strategy in strategies:
                try:
                    # Log which instrument we're processing
                    if current_time.second % 10 == 0:  # Every 10 seconds
                        # Get current price for this instrument
                        tick = mt5.symbol_info_tick(strategy.symbol)
                        if tick:
                            logger.info(
                                f"[{ny_time.strftime('%H:%M:%S')} NY] Processing {strategy.instrument_name} - Current bid: {tick.bid:.5f}, ask: {tick.ask:.5f}")
                            active_count += 1

                    strategy.run_iteration(current_time)
                except Exception as e:
                    logger.error(
                        f"[{ny_time.strftime('%H:%M:%S')} NY] Error running strategy for {strategy.instrument_name}: {e}",
                        exc_info=True)
                    # Log full traceback for debugging
                    import traceback
                    logger.error(traceback.format_exc())

            # Log every 10 seconds how many instruments are active
            if current_time.second % 10 == 0:
                logger.info(f"[{ny_time.strftime('%H:%M:%S')} NY] Active instruments: {active_count}/{len(strategies)}")

            # Log status every 15 minutes on the minute
            if current_time.minute % 15 == 0 and current_time.second < 5:
                # Get account information
                account_info = mt5.account_info()
                if account_info:
                    logger.info(
                        f"[{ny_time.strftime('%H:%M:%S')} NY] Account balance: {account_info.balance}, equity: {account_info.equity}")

            # Sleep to avoid excessive CPU usage
            time.sleep(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        # Clean up
        logger.info("Cleaning up...")

        # Close strategies
        for strategy in strategies:
            try:
                strategy.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup for {strategy.instrument_name}: {e}")

        # Disconnect from MT5
        mt5.shutdown()

        logger.info("Bot stopped")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Silver Bullet Trading Bot")
    parser.add_argument("--broker", type=str, help="Broker name from configuration")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug mode
    if args.debug:
        LOG_LEVEL = "DEBUG"

    # Run the bot
    run_bot(args.broker)