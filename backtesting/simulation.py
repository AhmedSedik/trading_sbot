# silver_bullet_bot/backtesting/simulation.py

import os
import logging
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import time

from silver_bullet_bot.config import INSTRUMENTS, ACTIVE_INSTRUMENTS
from backtesting.backtest_engine import BacktestEngine
from backtesting.reporting import ReportGenerator


class BacktestSimulation:
    """
    Runs backtest simulations for ICT Silver Bullet strategy
    """

    def __init__(self, output_dir='backtest_results', logger=None):
        """
        Initialize the backtest simulation

        Parameters:
        -----------
        output_dir : str
            Directory for storing backtest results
        logger : logging.Logger
            Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or self._setup_logger()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize backtest engine and report generator
        self.engine = BacktestEngine(output_dir=output_dir, logger=self.logger)
        self.report_generator = ReportGenerator(output_dir=output_dir, logger=self.logger)

    def _setup_logger(self):
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

    def run_simulation(self, instruments=None, start_date=None, end_date=None, initial_balance=10000,
                       generate_reports=True, report_formats=None, visualize=True):
        """
        Run backtest simulation for specified instruments and date range

        Parameters:
        -----------
        instruments : list or str
            List of instruments to test, or 'all' for all instruments
        start_date : str or datetime
            Start date for backtest
        end_date : str or datetime
            End date for backtest
        initial_balance : float
            Initial account balance
        generate_reports : bool
            Whether to generate reports
        report_formats : list
            List of report formats to generate
        visualize : bool
            Whether to generate visualizations

        Returns:
        --------
        dict
            Dictionary with backtest results for each instrument
        """
        # Set default report formats
        if report_formats is None:
            report_formats = ['csv', 'json', 'html']

        # Handle start and end dates
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')

        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Determine instruments to test
        if instruments is None or instruments == []:
            instruments = ACTIVE_INSTRUMENTS
        elif instruments == 'all':
            instruments = [inst for inst in INSTRUMENTS if inst != 'defaults']
        elif isinstance(instruments, str):
            instruments = [instruments]

        # Validate instruments
        valid_instruments = []
        for inst in instruments:
            if inst in INSTRUMENTS:
                valid_instruments.append(inst)
            else:
                self.logger.warning(f"Instrument {inst} not found in configuration, skipping")

        if not valid_instruments:
            self.logger.error("No valid instruments specified")
            return {}

        self.logger.info(f"Starting backtest simulation for {len(valid_instruments)} instruments")
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Initial balance: ${initial_balance:.2f}")

        # Run backtests
        results = {}
        all_results = []  # For aggregate reporting

        overall_start_time = time.time()

        for i, instrument in enumerate(valid_instruments, 1):
            self.logger.info(f"[{i}/{len(valid_instruments)}] Running backtest for {instrument}")

            try:
                # Run backtest
                result = self.engine.run_backtest(
                    instrument_name=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    balance=initial_balance
                )

                if result:
                    results[instrument] = result
                    all_results.append(result)

                    # Generate reports
                    if generate_reports:
                        self.report_generator.generate_reports(
                            result, report_formats=report_formats
                        )

            except Exception as e:
                self.logger.error(f"Error running backtest for {instrument}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Calculate overall execution time
        overall_execution_time = time.time() - overall_start_time
        self.logger.info(f"All backtests completed in {overall_execution_time:.2f} seconds")

        # Generate aggregate report if more than one instrument
        if len(results) > 1:
            self._generate_aggregate_report(results, report_formats)

        return results

    def _generate_aggregate_report(self, results, report_formats):
        """
        Generate aggregate report for multiple instruments

        Parameters:
        -----------
        results : dict
            Dictionary with backtest results for each instrument
        report_formats : list
            List of report formats to generate
        """
        self.logger.info("Generating aggregate report for all instruments")

        # Create summary dataframe
        summary_data = []

        for instrument, result in results.items():
            summary_data.append({
                'instrument': instrument,
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'net_profit': result.get('net_profit', 0),
                'percent_return': result.get('percent_return', 0),
                'profit_factor': result.get('profit_factor', 0),
                'avg_r_multiple': result.get('avg_r_multiple', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'final_balance': result.get('balance', 0)
            })

        if not summary_data:
            self.logger.warning("No results to include in aggregate report")
            return

        summary_df = pd.DataFrame(summary_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to CSV
        if 'csv' in report_formats:
            csv_file = os.path.join(self.output_dir, f"aggregate_summary_{timestamp}.csv")
            summary_df.to_csv(csv_file, index=False)
            self.logger.info(f"Aggregate summary saved to {csv_file}")

        # Generate visualizations
        charts_dir = os.path.join(self.output_dir, 'reports', 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        # Plot returns chart
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['instrument'], summary_df['percent_return'])
        plt.title('Return by Instrument (%)')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        returns_chart = os.path.join(charts_dir, f"aggregate_returns_{timestamp}.png")
        plt.savefig(returns_chart)
        plt.close()

        # Plot win rate chart
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['instrument'], summary_df['win_rate'])
        plt.title('Win Rate by Instrument')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        winrate_chart = os.path.join(charts_dir, f"aggregate_winrate_{timestamp}.png")
        plt.savefig(winrate_chart)
        plt.close()

        # Plot trade count chart
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['instrument'], summary_df['total_trades'])
        plt.title('Trade Count by Instrument')
        plt.ylabel('Number of Trades')
        plt.xticks(rotation=45)
        plt.tight_layout()
        trades_chart = os.path.join(charts_dir, f"aggregate_trades_{timestamp}.png")
        plt.savefig(trades_chart)
        plt.close()

        self.logger.info(f"Aggregate visualizations saved to {charts_dir}")