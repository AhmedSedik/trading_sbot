# silver_bullet_bot/visualize.py
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union

# Local imports

from data_loader import DataLoader
def setup_logger():
    """Set up and return a logger"""
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console)

    return logger


class StrategyVisualizer:
    """Class for visualizing trading strategy and backtest results"""

    def __init__(self, data_dir: str = "data", results_dir: str = "backtest_results", logger=None):
        """
        Initialize the visualizer

        Parameters:
        -----------
        data_dir : str
            Directory containing data files
        results_dir : str
            Directory containing backtest results
        logger : logging.Logger
            Logger instance
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.logger = logger or setup_logger()
        self.data_loader = DataLoader(data_dir, self.logger)

        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_context("paper")

    def load_backtest_results(self, result_file: str) -> Dict:
        """
        Load backtest results from JSON file

        Parameters:
        -----------
        result_file : str
            Path to result file

        Returns:
        --------
        Dict
            Dictionary containing backtest results
        """
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            self.logger.error(f"Error loading backtest results: {e}")
            return None

    def find_latest_backtest(self, symbol: str = None) -> str:
        """
        Find the latest backtest result file

        Parameters:
        -----------
        symbol : str
            Symbol to find results for

        Returns:
        --------
        str
            Path to latest result file
        """
        result_files = []

        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue

            if not filename.startswith('backtest_'):
                continue

            if symbol and symbol.lower() not in filename.lower():
                continue

            file_path = os.path.join(self.results_dir, filename)
            result_files.append((file_path, os.path.getmtime(file_path)))

        if not result_files:
            return None

        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: x[1], reverse=True)
        return result_files[0][0]

    def plot_trades_on_price(self,
                             symbol: str,
                             timeframe: str,
                             trades: List[Dict],
                             start_date: Union[str, datetime] = None,
                             end_date: Union[str, datetime] = None,
                             output_file: str = None) -> None:
        """
        Plot trades on price chart

        Parameters:
        -----------
        symbol : str
            Symbol to plot
        timeframe : str
            Timeframe to plot
        trades : List[Dict]
            List of trades to plot
        start_date : str or datetime
            Start date for chart
        end_date : str or datetime
            End date for chart
        output_file : str
            Output file path
        """
        # Load price data
        data = self.data_loader.load_multi_timeframe_data(symbol, [timeframe], start_date, end_date)

        if timeframe not in data or data[timeframe].empty:
            self.logger.error(f"No data available for {symbol} on {timeframe}")
            return

        df = data[timeframe]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price
        ax.plot(df['time'], df['close'], label='Price', color='#1f77b4', linewidth=1)

        # Add candlesticks (optional)
        if len(df) <= 200:  # Only show candlesticks for smaller datasets
            for i, row in df.iterrows():
                color = 'green' if row['is_bullish'] else 'red'
                ax.plot([row['time'], row['time']], [row['low'], row['high']], color=color, linewidth=0.8)

                # Draw body
                width = timedelta(days=0.6)  # Adjust width based on timeframe
                if timeframe in ['M1', 'M3', 'M5', 'M15']:
                    width = timedelta(minutes=0.6)
                elif timeframe in ['H1', 'H4']:
                    width = timedelta(hours=0.6)

                rect = Rectangle(
                    (row['time'] - width / 2, min(row['open'], row['close'])),
                    width,
                    abs(row['close'] - row['open']),
                    color=color,
                    alpha=0.5
                )
                ax.add_patch(rect)

        # Plot trades
        for trade in trades:
            try:
                # Convert trade times from string to datetime if needed
                entry_time = trade['entry_time']
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

                exit_time = trade.get('exit_time')
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))

                # Plot entry
                ax.scatter(entry_time, trade['entry'],
                           marker='^' if trade['trade_type'] == 'BUY' else 'v',
                           color='green' if trade['trade_type'] == 'BUY' else 'red',
                           s=100, label='_nolegend_')

                # Plot exit if available
                if exit_time and trade.get('exit_price'):
                    ax.scatter(exit_time, trade['exit_price'],
                               marker='o',
                               color='blue' if trade.get('pnl', 0) > 0 else 'orange',
                               s=100, label='_nolegend_')

                    # Plot trade line
                    ax.plot([entry_time, exit_time], [trade['entry'], trade['exit_price']],
                            color='green' if trade.get('pnl', 0) > 0 else 'red',
                            linestyle='--', linewidth=1)

                # Plot SL/TP levels
                if 'sl' in trade:
                    ax.axhline(y=trade['sl'], color='red', linestyle=':', linewidth=1,
                               xmin=(entry_time - df['time'].min()).total_seconds() / (
                                           df['time'].max() - df['time'].min()).total_seconds(),
                               xmax=(exit_time - df['time'].min()).total_seconds() / (
                                           df['time'].max() - df['time'].min()).total_seconds() if exit_time else 1)

                if 'tp' in trade:
                    ax.axhline(y=trade['tp'], color='green', linestyle=':', linewidth=1,
                               xmin=(entry_time - df['time'].min()).total_seconds() / (
                                           df['time'].max() - df['time'].min()).total_seconds(),
                               xmax=(exit_time - df['time'].min()).total_seconds() / (
                                           df['time'].max() - df['time'].min()).total_seconds() if exit_time else 1)

                # Annotate trade result
                if trade.get('pnl') is not None:
                    label = f"{trade.get('status', '')}: {trade.get('pnl', 0):.2f}"
                    if exit_time:
                        ax.annotate(label, (exit_time, trade['exit_price']),
                                    xytext=(10, 0), textcoords='offset points')
            except Exception as e:
                self.logger.error(f"Error plotting trade: {e}")
                continue

        # Formatting
        ax.set_title(f'{symbol} - {timeframe} with Trades')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')

        # Format x-axis based on timeframe
        if timeframe in ['M1', 'M3', 'M5', 'M15']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif timeframe in ['H1', 'H4']:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved trade visualization to {output_file}")
        else:
            plt.show()

    def plot_equity_curve(self,
                          equity_curve: List[float],
                          title: str = "Equity Curve",
                          output_file: str = None) -> None:
        """
        Plot equity curve

        Parameters:
        -----------
        equity_curve : List[float]
            List of equity values
        title : str
            Plot title
        output_file : str
            Output file path
        """
        if not equity_curve:
            self.logger.error("No equity curve data provided")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert to numpy array
        equity_array = np.array(equity_curve)

        # Plot equity curve
        ax.plot(equity_array, label='Equity', color='blue', linewidth=2)

        # Plot high watermark
        high_watermark = np.maximum.accumulate(equity_array)
        ax.plot(high_watermark, label='High Watermark', color='green', linestyle='--', linewidth=1)

        # Plot drawdown
        drawdown = high_watermark - equity_array
        ax.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red', label='Drawdown')

        # Formatting
        ax.set_title(title)
        ax.set_xlabel('Bars')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate and display max drawdown
        max_drawdown = drawdown.max()
        max_dd_pct = (max_drawdown / high_watermark[np.argmax(drawdown)]) * 100 if high_watermark[
                                                                                       np.argmax(drawdown)] > 0 else 0
        ax.text(0.02, 0.05, f'Max Drawdown: ${max_drawdown:.2f} ({max_dd_pct:.2f}%)',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved equity curve to {output_file}")
        else:
            plt.show()

    def plot_trade_distribution(self,
                                trades: List[Dict],
                                title: str = "Trade P/L Distribution",
                                output_file: str = None) -> None:
        """
        Plot distribution of trade profits and losses

        Parameters:
        -----------
        trades : List[Dict]
            List of trades
        title : str
            Plot title
        output_file : str
            Output file path
        """
        if not trades:
            self.logger.error("No trades data provided")
            return

        # Extract PnL values
        pnls = [trade.get('pnl', 0) for trade in trades if trade.get('pnl') is not None]

        if not pnls:
            self.logger.error("No PnL data found in trades")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create histogram
        sns.histplot(pnls, bins=20, kde=True, ax=ax)

        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)

        # Formatting
        ax.set_title(title)
        ax.set_xlabel('Profit/Loss')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Calculate and display statistics
        win_count = sum(1 for p in pnls if p > 0)
        loss_count = sum(1 for p in pnls if p <= 0)
        win_rate = win_count / len(pnls) if pnls else 0
        avg_win = sum(p for p in pnls if p > 0) / win_count if win_count else 0
        avg_loss = sum(p for p in pnls if p <= 0) / loss_count if loss_count else 0

        stats_text = (
            f'Win Rate: {win_rate:.2%}\n'
            f'Win Count: {win_count}\n'
            f'Loss Count: {loss_count}\n'
            f'Avg Win: ${avg_win:.2f}\n'
            f'Avg Loss: ${avg_loss:.2f}'
        )

        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved trade distribution to {output_file}")
        else:
            plt.show()

    def visualize_backtest(self, result_file: str = None, symbol: str = None) -> None:
        """
        Visualize backtest results

        Parameters:
        -----------
        result_file : str
            Path to result file
        symbol : str
            Symbol to find results for if no file specified
        """
        # Find result file if not specified
        if not result_file:
            result_file = self.find_latest_backtest(symbol)

        if not result_file:
            self.logger.error(f"No backtest results found for {symbol}")
            return

        # Load results
        results = self.load_backtest_results(result_file)

        if not results:
            return

        # Create output directory
        output_dir = os.path.join(self.results_dir, "visuals")
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename without extension
        base_filename = os.path.splitext(os.path.basename(result_file))[0]

        # Plot equity curve
        if 'equity_curve' in results:
            self.plot_equity_curve(
                results['equity_curve'],
                f"Equity Curve - {results.get('symbol', 'Unknown')}",
                os.path.join(output_dir, f"{base_filename}_equity.png")
            )

        # Plot trade distribution
        if 'trades' in results:
            self.plot_trade_distribution(
                results['trades'],
                f"Trade P/L Distribution - {results.get('symbol', 'Unknown')}",
                os.path.join(output_dir, f"{base_filename}_distribution.png")
            )

        # Plot trades on price chart (use primary timeframe)
        if 'trades' in results and 'symbol' in results:
            # Try to find the most suitable timeframe
            primary_timeframes = ['M1', 'M3', 'M5', 'M15', 'H1']
            available_timeframes = self.data_loader.get_available_timeframes(results['symbol'])

            # Find the first available timeframe in order of preference
            timeframe = next((tf for tf in primary_timeframes if tf in available_timeframes), None)

            if timeframe:
                # Get start and end dates with some padding
                start_date = min([trade['entry_time'] for trade in results['trades']
                                  if 'entry_time' in trade], default=None)
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                end_date = max([trade.get('exit_time', trade['entry_time']) for trade in results['trades']
                                if 'entry_time' in trade], default=None)
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Add padding
                if start_date:
                    start_date = start_date - timedelta(days=1)
                if end_date:
                    end_date = end_date + timedelta(days=1)

                self.plot_trades_on_price(
                    results['symbol'],
                    timeframe,
                    results['trades'],
                    start_date,
                    end_date,
                    os.path.join(output_dir, f"{base_filename}_trades.png")
                )

        print(f"Visualizations saved to {output_dir}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Visualize Silver Bullet trading strategy results')
    parser.add_argument('-f', '--file', help='Backtest result file to visualize')
    parser.add_argument('-s', '--symbol', help='Symbol to find results for if no file specified')
    parser.add_argument('-d', '--data-dir', default='data', help='Directory containing data files')
    parser.add_argument('-r', '--results-dir', default='backtest_results', help='Directory containing result files')

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Create visualizer
    visualizer = StrategyVisualizer(args.data_dir, args.results_dir, logger)

    # Visualize backtest
    visualizer.visualize_backtest(args.file, args.symbol)


if __name__ == "__main__":
    main()