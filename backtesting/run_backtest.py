# silver_bullet_bot/backtesting/run_backtest.py

import argparse
import logging
from datetime import datetime

from backtesting.simulation import BacktestSimulation


def setup_logger():
    """Set up and return a logger"""
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.INFO)

    # Ensure we don't add duplicate handlers
    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)

        # Add handler
        logger.addHandler(console)

    return logger


def main():
    """Main entry point for running backtests"""
    parser = argparse.ArgumentParser(description='Run Silver Bullet strategy backtest')

    # Instrument selection
    instrument_group = parser.add_mutually_exclusive_group()
    instrument_group.add_argument('-i', '--instrument', help='Specific instrument to test')
    instrument_group.add_argument('-a', '--all', action='store_true', help='Test all instruments')

    # Date range
    parser.add_argument('-s', '--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('-e', '--end-date', help='End date (YYYY-MM-DD)')

    # Other parameters
    parser.add_argument('-b', '--initial-balance', type=float, default=100000, help='Initial account balance')
    parser.add_argument('-o', '--output-dir', default='backtest_results', help='Output directory')
    parser.add_argument('-v', '--visualize', action='store_true', default=True, help='Generate visualizations')
    parser.add_argument('-r', '--report-formats', nargs='+', choices=['csv', 'json', 'html'],
                        default=['csv', 'json', 'html'], help='Report formats to generate')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set up logger
    logger = setup_logger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # Determine instruments to test
    instruments = None
    if args.all:
        instruments = 'all'
    elif args.instrument:
        instruments = [args.instrument]

    # Create simulation instance
    simulation = BacktestSimulation(output_dir=args.output_dir, logger=logger)

    # Run simulation
    simulation.run_simulation(
        instruments=instruments,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        generate_reports=True,
        report_formats=args.report_formats,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()