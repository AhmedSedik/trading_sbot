# silver_bullet_bot/backtesting/reporting.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jinja2 import Template, Environment, FileSystemLoader
import logging
import traceback
from datetime import datetime, time
import json

# Add this class after the imports
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):  # Handle datetime, date, and time objects
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        # Add this line to handle Period objects
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif hasattr(obj, 'to_dict'):  # Handle pandas objects
            return None  # Skip pandas objects that can't be serialized
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        return super().default(obj)


class ReportGenerator:
    """
    Generates detailed reports from backtest results
    """

    def __init__(self, output_dir='backtest_results', logger=None):
        """
        Initialize the report generator

        Parameters:
        -----------
        output_dir : str
            Directory for storing report output
        logger : logging.Logger
            Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger('backtest')

        # Create reports directory
        self.reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)

        # Set up Jinja2 environment for HTML templates
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir, exist_ok=True)
            self._create_default_templates(templates_dir)

        self.jinja_env = Environment(loader=FileSystemLoader(templates_dir))

    def generate_reports(self, backtest_result, report_formats=None):
        """
        Generate all requested report formats

        Parameters:
        -----------
        backtest_result : dict
            Backtest results from BacktestEngine
        report_formats : list
            List of report formats to generate ('csv', 'json', 'html')

        Returns:
        --------
        dict
            Dictionary with paths to generated reports
        """
        if not backtest_result:
            self.logger.error("No backtest results provided")
            return {}

        if report_formats is None:
            report_formats = ['csv', 'json', 'html']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instrument = backtest_result.get('instrument', 'unknown')

        generated_files = {}

        try:
            # Prepare trades data
            trades_df = self._prepare_trades_dataframe(backtest_result)

            # Prepare daily stats
            daily_stats_df = self._prepare_daily_stats(backtest_result)

            # Generate reports for each format
            for fmt in report_formats:
                if fmt == 'csv':
                    try:
                        # Trade journal CSV
                        trades_file = os.path.join(self.reports_dir, f"{instrument}_trades_{timestamp}.csv")
                        trades_df.to_csv(trades_file, index=False)
                        generated_files['trades_csv'] = trades_file

                        # Daily stats CSV
                        daily_file = os.path.join(self.reports_dir, f"{instrument}_daily_{timestamp}.csv")
                        daily_stats_df.to_csv(daily_file, index=True)
                        generated_files['daily_csv'] = daily_file

                        # Monthly stats CSV
                        monthly_df = self._prepare_monthly_stats(trades_df)
                        monthly_file = os.path.join(self.reports_dir, f"{instrument}_monthly_{timestamp}.csv")
                        monthly_df.to_csv(monthly_file, index=True)
                        generated_files['monthly_csv'] = monthly_file

                        self.logger.info(f"CSV reports generated successfully")
                    except Exception as e:
                        self.logger.error(f"Error generating CSV reports: {str(e)}")

                elif fmt == 'json':
                    try:
                        # Full backtest results JSON
                        backtest_copy = self._prepare_for_json(backtest_result)
                        json_file = os.path.join(self.reports_dir, f"{instrument}_backtest_{timestamp}.json")
                        with open(json_file, 'w') as f:
                            json.dump(backtest_copy, f, indent=2, cls=CustomJSONEncoder)
                        generated_files['backtest_json'] = json_file

                        # Trade journal JSON - handle manually to avoid recursion errors
                        trades_file = os.path.join(self.reports_dir, f"{instrument}_trades_{timestamp}.json")
                        # Convert dataframe to simpler format first
                        trades_list = []
                        for _, row in trades_df.iterrows():
                            trade_dict = {}
                            for col in trades_df.columns:
                                # Handle special types
                                if pd.api.types.is_datetime64_dtype(trades_df[col]):
                                    trade_dict[col] = row[col].isoformat() if pd.notnull(row[col]) else None
                                elif isinstance(row[col], pd.Timedelta):
                                    trade_dict[col] = str(row[col])
                                elif hasattr(row[col], 'to_dict'):
                                    # Handle nested pandas objects
                                    trade_dict[col] = None
                                else:
                                    trade_dict[col] = row[col]
                            trades_list.append(trade_dict)

                        # Write to JSON manually
                        with open(trades_file, 'w') as f:
                            json.dump(trades_list, f, indent=2, cls=CustomJSONEncoder)
                        generated_files['trades_json'] = trades_file

                        self.logger.info(f"JSON reports generated successfully")
                    except Exception as e:
                        self.logger.error(f"Error generating JSON reports: {str(e)}")
                        self.logger.error(traceback.format_exc())

                elif fmt == 'html':
                    try:
                        # Generate HTML report
                        html_file = self._generate_html_report(
                            backtest_result, trades_df, daily_stats_df, timestamp
                        )
                        if html_file:
                            generated_files['report_html'] = html_file
                    except Exception as e:
                        self.logger.error(f"Error generating HTML report: {str(e)}")
                        self.logger.error(traceback.format_exc())

            self.logger.info(f"Generated reports: {list(generated_files.keys())}")
            return generated_files

        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")
            self.logger.error(traceback.format_exc())
            return generated_files

    def _prepare_trades_dataframe(self, backtest_result):
        """Prepare trades dataframe for reporting"""
        if 'trades' not in backtest_result or not backtest_result['trades']:
            self.logger.warning("No trades found in backtest results")
            return pd.DataFrame()

        # Extract trades and convert to dataframe
        trades = backtest_result['trades']

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Convert datetime strings to datetime objects
        for col in ['entry_time', 'exit_time']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    self.logger.warning(f"Error converting {col} to datetime: {e}")

        # Calculate trade duration for closed trades
        if 'entry_time' in df.columns and 'exit_time' in df.columns and 'status' in df.columns:
            try:
                # Only calculate for closed trades
                closed_mask = df['status'] == 'closed'
                if closed_mask.any():
                    df.loc[closed_mask, 'duration'] = df.loc[closed_mask, 'exit_time'] - df.loc[
                        closed_mask, 'entry_time']
                    # Convert timedelta to minutes
                    df.loc[closed_mask, 'duration_minutes'] = df.loc[closed_mask, 'duration'].dt.total_seconds() / 60
            except Exception as e:
                self.logger.warning(f"Error calculating trade duration: {e}")

        # Extract date and time info
        if 'entry_time' in df.columns:
            df['date'] = df['entry_time'].dt.date
            df['day_of_week'] = df['entry_time'].dt.day_name()
            df['hour'] = df['entry_time'].dt.hour

        # Create display columns for HTML rendering (as strings)
        # This is the key change - create separate string columns for display
        if 'status' in df.columns:
            closed_mask = df['status'] == 'closed'
            self.logger.info(f"closed_mask: {closed_mask}")

            # Check for closed trades with missing exit data
            if closed_mask.any():
                # Check for missing exit times
                if 'exit_time' in df.columns:
                    missing_exit_time = df['exit_time'].isna() & closed_mask
                    if missing_exit_time.any():
                        self.logger.warning(f"Found {missing_exit_time.sum()} closed trades with missing exit times")
                        # For trades with missing exit times, use the entry time + 1 day as a fallback
                        df.loc[missing_exit_time, 'exit_time'] = df.loc[missing_exit_time, 'entry_time'] + pd.Timedelta(
                            days=1)

                # Check for missing exit prices
                if 'exit_price' in df.columns:
                    missing_exit_price = df['exit_price'].isna() & closed_mask
                    if missing_exit_price.any():
                        self.logger.warning(f"Found {missing_exit_price.sum()} closed trades with missing exit prices")
                        # For trades with missing exit prices, use entry price as a fallback (0 profit)
                        df.loc[missing_exit_price, 'exit_price'] = df.loc[missing_exit_price, 'entry_price']

                # Check for missing exit reasons
                if 'exit_reason' in df.columns:
                    missing_exit_reason = df['exit_reason'].isna() & closed_mask
                    if missing_exit_reason.any():
                        self.logger.warning(
                            f"Found {missing_exit_reason.sum()} closed trades with missing exit reasons")
                        df.loc[missing_exit_reason, 'exit_reason'] = 'unknown_exit'

                # Check for missing PnL values
                if 'pnl' in df.columns:
                    missing_pnl = df['pnl'].isna() & closed_mask
                    if missing_pnl.any():
                        self.logger.warning(f"Found {missing_pnl.sum()} closed trades with missing PnL values")
                        # Recalculate PnL for these trades
                        for idx in df.index[missing_pnl]:
                            trade = df.loc[idx]
                            direction = trade['direction']
                            entry_price = trade['entry_price']
                            exit_price = trade['exit_price']

                            if pd.notna(entry_price) and pd.notna(exit_price):
                                point_value = 1.0  # Default fallback point value
                                lot_size = trade.get('lot_size', 1.0)

                                if direction == 'buy':
                                    profit_points = exit_price - entry_price
                                else:
                                    profit_points = entry_price - exit_price

                                pnl = profit_points * point_value * lot_size
                                df.loc[idx, 'pnl'] = pnl

            if 'r_multiple' in df.columns:
                # Format R-multiples with 2 decimal places for closed trades
                df['r_multiple_str'] = df['r_multiple'].apply(
                    lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                # For open trades, set to "Pending"
                df.loc[df['status'] == 'open', 'r_multiple_str'] = "Pending"

        return df

    def _prepare_daily_stats(self, backtest_result):
        """
        Prepare daily statistics dataframe

        Parameters:
        -----------
        backtest_result : dict
            Backtest results

        Returns:
        --------
        pd.DataFrame
            Dataframe with daily statistics
        """
        # If date_stats exists, use it directly
        if 'date_stats' in backtest_result and backtest_result['date_stats']:
            try:
                # Convert dict to dataframe
                df = pd.DataFrame.from_dict(backtest_result['date_stats'], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                # Calculate additional metrics
                if 'wins' in df.columns and 'trades' in df.columns:
                    df['win_rate'] = df['wins'] / df['trades']
                    df['win_rate'] = df['win_rate'].fillna(0)

                # Calculate cumulative metrics
                if 'profit' in df.columns:
                    df['cumulative_profit'] = df['profit'].cumsum()

                return df

            except Exception as e:
                self.logger.warning(f"Error preparing daily stats from date_stats: {e}")

        # If no date_stats or error occurred, compute from trades
        if 'trades' in backtest_result and backtest_result['trades']:
            try:
                # Get trades dataframe
                trades_df = self._prepare_trades_dataframe(backtest_result)

                if trades_df.empty:
                    return pd.DataFrame()

                # Group by date
                daily_df = trades_df.groupby('date').agg({
                    'pnl': ['sum', 'count'],
                    'r_multiple': ['mean', 'sum'],
                })

                # Flatten multi-level columns
                daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns.values]

                # Rename columns
                daily_df = daily_df.rename(columns={
                    'pnl_sum': 'profit',
                    'pnl_count': 'trades',
                    'r_multiple_mean': 'avg_r',
                    'r_multiple_sum': 'total_r'
                })

                # Count winning trades
                wins = trades_df[trades_df['pnl'] > 0].groupby('date').size()
                daily_df['wins'] = wins
                daily_df['wins'] = daily_df['wins'].fillna(0)

                # Calculate win rate
                daily_df['win_rate'] = daily_df['wins'] / daily_df['trades']

                # Calculate cumulative metrics
                daily_df['cumulative_profit'] = daily_df['profit'].cumsum()

                return daily_df

            except Exception as e:
                self.logger.warning(f"Error preparing daily stats from trades: {e}")

        # If all else fails, return empty dataframe
        return pd.DataFrame()

    def _prepare_monthly_stats(self, trades_df):
        """Prepare monthly statistics dataframe"""
        if trades_df.empty:
            return pd.DataFrame()

        try:
            # Only use closed trades for statistics
            closed_trades = trades_df[trades_df['status'] == 'closed'].copy()

            if closed_trades.empty:
                return pd.DataFrame()

            # Add month and year columns
            if 'entry_time' in closed_trades.columns:
                # First ensure entry_time is timezone-naive by converting to datetime64[ns]
                closed_trades['entry_time_naive'] = closed_trades['entry_time'].astype('datetime64[ns]')
                # Then create the period
                closed_trades['year_month'] = closed_trades['entry_time_naive'].dt.to_period('M')
                # Drop the temporary column
                closed_trades = closed_trades.drop(columns=['entry_time_naive'])

            # Group by year-month
            monthly_df = closed_trades.groupby('year_month').agg({
                'pnl': ['sum', 'count'],
                'r_multiple': ['mean', 'sum'],
            })

            # Flatten multi-level columns
            monthly_df.columns = ['_'.join(col).strip() for col in monthly_df.columns.values]

            # Rename columns
            monthly_df = monthly_df.rename(columns={
                'pnl_sum': 'profit',
                'pnl_count': 'trades',
                'r_multiple_mean': 'avg_r',
                'r_multiple_sum': 'total_r'
            })

            # Count winning trades
            wins = closed_trades[closed_trades['pnl'] > 0].groupby('year_month').size()
            monthly_df['wins'] = wins
            monthly_df['wins'] = monthly_df['wins'].fillna(0)

            # Calculate win rate
            monthly_df['win_rate'] = monthly_df['wins'] / monthly_df['trades']

            # Calculate cumulative metrics
            monthly_df['cumulative_profit'] = monthly_df['profit'].cumsum()

            return monthly_df

        except Exception as e:
            self.logger.warning(f"Error preparing monthly stats: {e}")
            return pd.DataFrame()

    def _prepare_for_json(self, backtest_result):
        """
        Prepare backtest results for JSON serialization
        """
        # Create a deep copy
        result_copy = {}

        for key, value in backtest_result.items():
            # Skip non-serializable objects or large data structures
            if key in ['data_loader', 'logger', 'current_position']:
                continue

            # Special handling for trades
            if key == 'trades':
                result_copy[key] = []
                for trade in value:
                    trade_copy = dict(trade)
                    for time_field in ['entry_time', 'exit_time']:
                        if time_field in trade_copy and trade_copy[time_field] is not None:
                            if isinstance(trade_copy[time_field], pd.Timestamp):
                                trade_copy[time_field] = trade_copy[time_field].isoformat()
                            elif hasattr(trade_copy[time_field], 'isoformat'):
                                trade_copy[time_field] = trade_copy[time_field].isoformat()
                    result_copy[key].append(trade_copy)
            else:
                result_copy[key] = value

        # Convert to JSON string and back to handle serialization
        try:
            json_str = json.dumps(result_copy, cls=CustomJSONEncoder)
            return json.loads(json_str)
        except TypeError as e:
            self.logger.warning(f"JSON serialization error: {e}. Attempting fallback.")

            # Fallback method - check for time objects
            for key, value in list(result_copy.items()):
                # Check if it's a time-like object without using isinstance
                if hasattr(value, 'hour') and hasattr(value, 'minute') and hasattr(value, 'second'):
                    result_copy[key] = f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"

            return result_copy

    def _generate_html_report(self, backtest_result, trades_df, daily_stats, timestamp):
        """
        Generate HTML report from backtest results

        Parameters:
        -----------
        backtest_result : dict
            Backtest results
        trades_df : pd.DataFrame
            Dataframe with trade information
        daily_stats : pd.DataFrame
            Dataframe with daily statistics
        timestamp : str
            Timestamp for filename

        Returns:
        --------
        str
            Path to generated HTML report
        """
        try:
            instrument = backtest_result.get('instrument', 'unknown')

            # Create charts directory
            charts_dir = os.path.join(self.reports_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)

            # Generate chart images
            try:
                chart_paths = self._generate_chart_images(
                    backtest_result, trades_df, daily_stats, charts_dir, instrument, timestamp
                )
            except Exception as chart_error:
                self.logger.error(f"Error generating chart images: {str(chart_error)}")
                self.logger.error(traceback.format_exc())
                chart_paths = {}

            # Prepare template data
            template_data = {
                'title': f"Backtest Report - {instrument}",
                'instrument': instrument,
                'timestamp': timestamp,
                'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'start_date': backtest_result.get('start_date', ''),
                'end_date': backtest_result.get('end_date', ''),
                'execution_time': backtest_result.get('execution_time', 0),
                'summary': {
                    'initial_balance': backtest_result.get('initial_balance', 0),
                    'final_balance': backtest_result.get('balance', 0),
                    'net_profit': backtest_result.get('net_profit', 0),
                    'percent_return': backtest_result.get('percent_return', 0),
                    'total_trades': backtest_result.get('total_trades', 0),
                    'win_rate': backtest_result.get('win_rate', 0),
                    'profit_factor': backtest_result.get('profit_factor', 0),
                    'avg_profit': backtest_result.get('avg_profit', 0),
                    'avg_loss': backtest_result.get('avg_loss', 0),
                    'avg_r_multiple': backtest_result.get('avg_r_multiple', 0),
                    'max_drawdown': backtest_result.get('max_drawdown', 0),
                    'max_drawdown_pct': backtest_result.get('max_drawdown_pct', 0),
                },
                'chart_paths': chart_paths,
            }

            # Before rendering the trade table, prepare a display version with the display columns
            display_df = trades_df.copy()

            # For open trades, create display versions of columns
            if 'status' in display_df.columns:
                open_mask = display_df['status'] == 'open'

                # Handle exit_time display for open trades
                if 'exit_time' in display_df.columns:
                    # Create a string version of exit_time
                    display_df['exit_time_display'] = display_df['exit_time'].astype(str)
                    # Set "Pending" for open trades
                    display_df.loc[open_mask, 'exit_time_display'] = "Pending"
                    # Replace the original column
                    display_df['exit_time'] = display_df['exit_time_display']
                    display_df = display_df.drop(columns=['exit_time_display'])

                # Handle exit_price display
                if 'exit_price' in display_df.columns:
                    # Convert to string to avoid type issues
                    display_df['exit_price_display'] = display_df['exit_price'].astype(str)
                    # Set "Pending" for open trades
                    display_df.loc[open_mask, 'exit_price_display'] = "Pending"
                    # Replace the original column
                    display_df['exit_price'] = display_df['exit_price_display']
                    display_df = display_df.drop(columns=['exit_price_display'])

                # Handle exit_reason display
                if 'exit_reason' in display_df.columns:
                    # Convert to string to avoid type issues
                    display_df['exit_reason_display'] = display_df['exit_reason'].astype(str)
                    # Set "Pending" for open trades
                    display_df.loc[open_mask, 'exit_reason_display'] = "Pending"
                    # Replace the original column
                    display_df['exit_reason'] = display_df['exit_reason_display']
                    display_df = display_df.drop(columns=['exit_reason_display'])

                # Handle pnl display
                if 'pnl' in display_df.columns:
                    # Convert to string to avoid type issues
                    display_df['pnl_display'] = display_df['pnl'].astype(str)
                    # Set "Pending" for open trades
                    display_df.loc[open_mask, 'pnl_display'] = "Pending"
                    # Replace the original column
                    display_df['pnl'] = display_df['pnl_display']
                    display_df = display_df.drop(columns=['pnl_display'])

                # Handle r_multiple display
                if 'r_multiple' in display_df.columns:
                    # Convert to string to avoid type issues
                    display_df['r_multiple_display'] = display_df['r_multiple'].astype(str)
                    # Set "Pending" for open trades
                    display_df.loc[open_mask, 'r_multiple_display'] = "Pending"
                    # Replace the original column
                    display_df['r_multiple'] = display_df['r_multiple_display']
                    display_df = display_df.drop(columns=['r_multiple_display'])

            # Safely generate HTML tables with error handling
            try:
                # Convert trades dataframe to HTML with limited rows if very large
                if len(display_df) > 1000:
                    self.logger.warning(
                        f"Large trade dataframe detected ({len(display_df)} rows). Limiting to 1000 rows.")
                    template_data['trades_table'] = display_df.head(1000).to_html(
                        classes='table table-striped table-bordered',
                        index=False,
                        na_rep="--"
                    )
                else:
                    template_data['trades_table'] = display_df.to_html(
                        classes='table table-striped table-bordered',
                        index=False,
                        na_rep="--"
                    )

                # Daily stats table
                if not daily_stats.empty:
                    template_data['daily_stats_table'] = daily_stats.to_html(
                        classes='table table-striped table-bordered'
                    )
                else:
                    template_data['daily_stats_table'] = "<p>No daily statistics available</p>"

            except Exception as table_error:
                self.logger.error(f"Error generating HTML tables: {str(table_error)}")
                self.logger.error(traceback.format_exc())
                template_data['trades_table'] = "<p>Error generating trades table</p>"
                template_data['daily_stats_table'] = "<p>Error generating daily stats table</p>"

            # Generate HTML from template
            try:
                template = self.jinja_env.get_template('report_template.html')
                html_content = template.render(**template_data)

                # Save HTML report
                html_file = os.path.join(self.reports_dir, f"{instrument}_report_{timestamp}.html")
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                self.logger.info(f"HTML report generated: {html_file}")
                return html_file
            except Exception as template_error:
                self.logger.error(f"Error generating HTML from template: {str(template_error)}")
                self.logger.error(traceback.format_exc())

                # Try a fallback simple HTML report
                try:
                    simple_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Simple Backtest Report - {instrument}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2 {{ color: #333; }}
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                        </style>
                    </head>
                    <body>
                        <h1>Backtest Report - {instrument}</h1>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                        <h2>Summary</h2>
                        <table>
                            <tr><td>Initial Balance:</td><td>${template_data['summary']['initial_balance']:.2f}</td></tr>
                            <tr><td>Final Balance:</td><td>${template_data['summary']['final_balance']:.2f}</td></tr>
                            <tr><td>Net Profit:</td><td>${template_data['summary']['net_profit']:.2f}</td></tr>
                            <tr><td>Total Trades:</td><td>{template_data['summary']['total_trades']}</td></tr>
                            <tr><td>Win Rate:</td><td>{template_data['summary']['win_rate'] * 100:.2f}%</td></tr>
                        </table>

                        <p>Note: This is a simplified report due to an error generating the full report.</p>
                    </body>
                    </html>
                    """

                    fallback_file = os.path.join(self.reports_dir, f"{instrument}_simple_report_{timestamp}.html")
                    with open(fallback_file, 'w', encoding='utf-8') as f:
                        f.write(simple_html)

                    self.logger.info(f"Simple HTML report generated as fallback: {fallback_file}")
                    return fallback_file
                except Exception as fallback_error:
                    self.logger.error(f"Failed to generate even a simple HTML report: {str(fallback_error)}")
                    self.logger.error(traceback.format_exc())
                    return None

        except Exception as e:
            self.logger.error(f"Error in HTML report generation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_chart_images(self, backtest_result, trades_df, daily_stats, charts_dir, instrument, timestamp):
        """
        Generate chart images for HTML report

        Parameters:
        -----------
        backtest_result : dict
            Backtest results
        trades_df : pd.DataFrame
            Dataframe with trade information
        daily_stats : pd.DataFrame
            Dataframe with daily statistics
        charts_dir : str
            Directory for storing chart images
        instrument : str
            Instrument name
        timestamp : str
            Timestamp for filenames

        Returns:
        --------
        dict
            Dictionary with paths to generated chart images
        """
        chart_paths = {}

        try:
            # Set plot style
            plt.style.use('seaborn-v0_8-darkgrid')

            # Filter out open trades before any chart generation
            # This prevents type errors when comparing strings with numbers
            closed_trades_df = trades_df[
                trades_df['status'] == 'closed'].copy() if 'status' in trades_df.columns else trades_df.copy()

            # Ensure numeric columns are actually numeric
            for col in ['pnl', 'r_multiple', 'exit_price']:
                if col in closed_trades_df.columns:
                    try:
                        closed_trades_df[col] = pd.to_numeric(closed_trades_df[col], errors='coerce')
                    except:
                        self.logger.warning(f"Could not convert '{col}' to numeric type")

            # 1. Equity Curve
            if 'equity_curve' in backtest_result and backtest_result['equity_curve']:
                equity_filename = f"{instrument}_equity_{timestamp}.png"
                equity_path = os.path.join(charts_dir, equity_filename)

                plt.figure(figsize=(12, 6))
                plt.plot(backtest_result['equity_curve'], linewidth=2)
                plt.title('Equity Curve')
                plt.xlabel('Bars')
                plt.ylabel('Equity')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(equity_path)
                plt.close()

                chart_paths['equity_curve'] = os.path.relpath(equity_path, self.reports_dir)

            # 2. Daily Profit
            if not daily_stats.empty and 'profit' in daily_stats.columns:
                daily_profit_filename = f"{instrument}_daily_profit_{timestamp}.png"
                daily_profit_path = os.path.join(charts_dir, daily_profit_filename)

                plt.figure(figsize=(12, 6))
                daily_stats['profit'].plot(kind='bar', color=daily_stats['profit'].map(lambda x: 'g' if x > 0 else 'r'))
                plt.title('Daily Profit')
                plt.xlabel('Date')
                plt.ylabel('Profit')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(daily_profit_path)
                plt.close()

                chart_paths['daily_profit'] = os.path.relpath(daily_profit_path, self.reports_dir)

            # 3. Cumulative Profit
            if not daily_stats.empty and 'cumulative_profit' in daily_stats.columns:
                cumulative_profit_filename = f"{instrument}_cumulative_profit_{timestamp}.png"
                cumulative_profit_path = os.path.join(charts_dir, cumulative_profit_filename)

                plt.figure(figsize=(12, 6))
                daily_stats['cumulative_profit'].plot(linewidth=2)
                plt.title('Cumulative Profit')
                plt.xlabel('Date')
                plt.ylabel('Profit')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(cumulative_profit_path)
                plt.close()

                chart_paths['cumulative_profit'] = os.path.relpath(cumulative_profit_path, self.reports_dir)

            # 4. Win Rate by Day of Week
            if not closed_trades_df.empty and 'day_of_week' in closed_trades_df.columns and 'pnl' in closed_trades_df.columns:
                dow_win_rate_filename = f"{instrument}_dow_win_rate_{timestamp}.png"
                dow_win_rate_path = os.path.join(charts_dir, dow_win_rate_filename)

                try:
                    # Ensure we have sufficient data
                    if len(closed_trades_df) >= 3:  # Minimum trades for meaningful grouping
                        # Group by day of week
                        dow_stats = closed_trades_df.groupby('day_of_week').agg({
                            'pnl': 'count',
                        }).rename(columns={'pnl': 'trades'})

                        # Count winning trades by day of week
                        winning_trades = closed_trades_df[closed_trades_df['pnl'] > 0]
                        if not winning_trades.empty:
                            wins = winning_trades.groupby('day_of_week').size()
                            dow_stats['wins'] = wins
                            dow_stats['wins'] = dow_stats['wins'].fillna(0)

                            # Calculate win rate
                            dow_stats['win_rate'] = dow_stats['wins'] / dow_stats['trades']

                            # Order days of week correctly
                            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            dow_stats = dow_stats.reindex([d for d in days_order if d in dow_stats.index])

                            # Only plot if we have data
                            if not dow_stats.empty and 'win_rate' in dow_stats.columns:
                                plt.figure(figsize=(12, 6))
                                dow_stats['win_rate'].plot(kind='bar', color='skyblue')
                                plt.title('Win Rate by Day of Week')
                                plt.xlabel('Day of Week')
                                plt.ylabel('Win Rate')
                                plt.ylim(0, 1)
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()
                                plt.savefig(dow_win_rate_path)
                                plt.close()

                                chart_paths['dow_win_rate'] = os.path.relpath(dow_win_rate_path, self.reports_dir)
                except Exception as e:
                    self.logger.warning(f"Error generating day of week win rate chart: {str(e)}")

            # 5. P/L Distribution
            if not closed_trades_df.empty and 'pnl' in closed_trades_df.columns:
                pnl_dist_filename = f"{instrument}_pnl_dist_{timestamp}.png"
                pnl_dist_path = os.path.join(charts_dir, pnl_dist_filename)

                try:
                    # Drop any NaN values that might have crept in
                    valid_pnl = closed_trades_df['pnl'].dropna()

                    if len(valid_pnl) >= 5:  # Minimum trades for meaningful histogram
                        plt.figure(figsize=(12, 6))
                        sns.histplot(valid_pnl, bins=min(20, len(valid_pnl) // 2), kde=True)
                        plt.axvline(x=0, color='r', linestyle='--')
                        plt.title('P/L Distribution')
                        plt.xlabel('Profit/Loss')
                        plt.ylabel('Frequency')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(pnl_dist_path)
                        plt.close()

                        chart_paths['pnl_distribution'] = os.path.relpath(pnl_dist_path, self.reports_dir)
                except Exception as e:
                    self.logger.warning(f"Error generating P/L distribution chart: {str(e)}")

            # 6. R-Multiple Distribution
            if not closed_trades_df.empty and 'r_multiple' in closed_trades_df.columns:
                r_dist_filename = f"{instrument}_r_dist_{timestamp}.png"
                r_dist_path = os.path.join(charts_dir, r_dist_filename)

                try:
                    # Drop any NaN values
                    valid_r = closed_trades_df['r_multiple'].dropna()

                    if len(valid_r) >= 5:  # Minimum trades for meaningful histogram
                        plt.figure(figsize=(12, 6))
                        sns.histplot(valid_r, bins=min(20, len(valid_r) // 2), kde=True)
                        plt.axvline(x=0, color='r', linestyle='--')
                        plt.title('R-Multiple Distribution')
                        plt.xlabel('R-Multiple')
                        plt.ylabel('Frequency')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(r_dist_path)
                        plt.close()

                        chart_paths['r_distribution'] = os.path.relpath(r_dist_path, self.reports_dir)
                except Exception as e:
                    self.logger.warning(f"Error generating R-multiple distribution chart: {str(e)}")

            # 7. Hourly Win Rate
            if not closed_trades_df.empty and 'hour' in closed_trades_df.columns and 'pnl' in closed_trades_df.columns:
                hourly_win_rate_filename = f"{instrument}_hourly_win_rate_{timestamp}.png"
                hourly_win_rate_path = os.path.join(charts_dir, hourly_win_rate_filename)

                try:
                    # Ensure we have sufficient data
                    if len(closed_trades_df) >= 5:  # Minimum trades for meaningful grouping
                        # Group by hour
                        hour_stats = closed_trades_df.groupby('hour').agg({
                            'pnl': 'count',
                        }).rename(columns={'pnl': 'trades'})

                        # Count winning trades by hour
                        winning_trades = closed_trades_df[closed_trades_df['pnl'] > 0]
                        if not winning_trades.empty:
                            wins = winning_trades.groupby('hour').size()
                            hour_stats['wins'] = wins
                            hour_stats['wins'] = hour_stats['wins'].fillna(0)

                            # Calculate win rate
                            hour_stats['win_rate'] = hour_stats['wins'] / hour_stats['trades']

                            # Sort by hour
                            hour_stats = hour_stats.sort_index()

                            # Only plot if we have data
                            if not hour_stats.empty and 'win_rate' in hour_stats.columns:
                                plt.figure(figsize=(12, 6))
                                hour_stats['win_rate'].plot(kind='bar', color='lightgreen')
                                plt.title('Win Rate by Hour of Day')
                                plt.xlabel('Hour')
                                plt.ylabel('Win Rate')
                                plt.ylim(0, 1)
                                plt.grid(True, alpha=0.3)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                plt.savefig(hourly_win_rate_path)
                                plt.close()

                                chart_paths['hourly_win_rate'] = os.path.relpath(hourly_win_rate_path, self.reports_dir)
                except Exception as e:
                    self.logger.warning(f"Error generating hourly win rate chart: {str(e)}")

            return chart_paths

        except Exception as e:
            self.logger.error(f"Error generating chart images: {str(e)}")
            self.logger.error(traceback.format_exc())
            return chart_paths

    def _create_default_templates(self, templates_dir):
        """
        Create default HTML templates

        Parameters:
        -----------
        templates_dir : str
            Directory for storing templates
        """
        # Create report template
        report_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                .chart-container { margin-bottom: 30px; }
                .table-responsive { margin-bottom: 30px; overflow-x: auto; }
                .metric-card { margin-bottom: 20px; }

                /* IMPROVED TABLE STYLING */
                table.table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }

                table.table th, 
                table.table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }

                table.table th {
                    background-color: #f2f2f2;
                    position: sticky;
                    top: 0;
                }

                table.table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }

                table.table tr:hover {
                    background-color: #f1f1f1;
                }

                /* Add horizontal scrolling for wide tables */
                .table-responsive {
                    max-width: 100%;
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ title }}</h1>
                <p class="text-muted">Generated on {{ run_date }}</p>

                <div class="card mb-4">
                    <div class="card-header">
                        <h3>Backtest Summary</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h4>Trade Statistics</h4>
                                <table class="table table-bordered">
                                    <tr>
                                        <td>Instrument:</td>
                                        <td>{{ instrument }}</td>
                                    </tr>
                                    <tr>
                                        <td>Period:</td>
                                        <td>{{ start_date }} to {{ end_date }}</td>
                                    </tr>
                                    <tr>
                                        <td>Total Trades:</td>
                                        <td>{{ summary.total_trades }}</td>
                                    </tr>
                                    <tr>
                                        <td>Win Rate:</td>
                                        <td>{{ "%.2f%%" | format(summary.win_rate * 100) }}</td>
                                    </tr>
                                    <tr>
                                        <td>Profit Factor:</td>
                                        <td>{{ "%.2f" | format(summary.profit_factor) }}</td>
                                    </tr>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h4>Performance</h4>
                                <table class="table table-bordered">
                                    <tr>
                                        <td>Initial Balance:</td>
                                        <td>${{ "%.2f" | format(summary.initial_balance) }}</td>
                                    </tr>
                                    <tr>
                                        <td>Final Balance:</td>
                                        <td>${{ "%.2f" | format(summary.final_balance) }}</td>
                                    </tr>
                                    <tr>
                                        <td>Net Profit:</td>
                                        <td>${{ "%.2f" | format(summary.net_profit) }} ({{ "%.2f%%" | format(summary.percent_return) }})</td>
                                    </tr>
                                    <tr>
                                        <td>Max Drawdown:</td>
                                        <td>${{ "%.2f" | format(summary.max_drawdown) }} ({{ "%.2f%%" | format(summary.max_drawdown_pct) }})</td>
                                    </tr>
                                    <tr>
                                        <td>Average R-Multiple:</td>
                                        <td>{{ "%.2f" | format(summary.avg_r_multiple) }}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <h3>Performance Charts</h3>
                <div class="row">
                    {% if chart_paths.equity_curve %}
                    <div class="col-md-6 chart-container">
                        <div class="card">
                            <div class="card-header">Equity Curve</div>
                            <div class="card-body">
                                <img src="{{ chart_paths.equity_curve }}" class="img-fluid" alt="Equity Curve">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if chart_paths.cumulative_profit %}
                    <div class="col-md-6 chart-container">
                        <div class="card">
                            <div class="card-header">Cumulative Profit</div>
                            <div class="card-body">
                                <img src="{{ chart_paths.cumulative_profit }}" class="img-fluid" alt="Cumulative Profit">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if chart_paths.daily_profit %}
                    <div class="col-md-6 chart-container">
                        <div class="card">
                            <div class="card-header">Daily Profit</div>
                            <div class="card-body">
                                <img src="{{ chart_paths.daily_profit }}" class="img-fluid" alt="Daily Profit">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if chart_paths.dow_win_rate %}
                    <div class="col-md-6 chart-container">
                        <div class="card">
                            <div class="card-header">Win Rate by Day of Week</div>
                            <div class="card-body">
                                <img src="{{ chart_paths.dow_win_rate }}" class="img-fluid" alt="Win Rate by Day of Week">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if chart_paths.pnl_distribution %}
                    <div class="col-md-6 chart-container">
                        <div class="card">
                            <div class="card-header">P/L Distribution</div>
                            <div class="card-body">
                                <img src="{{ chart_paths.pnl_distribution }}" class="img-fluid" alt="P/L Distribution">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if chart_paths.r_distribution %}
                    <div class="col-md-6 chart-container">
                        <div class="card">
                            <div class="card-header">R-Multiple Distribution</div>
                            <div class="card-body">
                                <img src="{{ chart_paths.r_distribution }}" class="img-fluid" alt="R-Multiple Distribution">
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>

                <h3>Trade Journal</h3>
                <div class="table-responsive">
                    {{ trades_table|safe }}
                </div>

                <h3>Daily Statistics</h3>
                <div class="table-responsive">
                    {{ daily_stats_table|safe }}
                </div>

                <footer class="my-5 pt-5 text-muted text-center text-small">
                    <p>Report generated by Silver Bullet Backtesting Engine</p>
                    <p>Execution time: {{ "%.2f" | format(execution_time) }} seconds</p>
                </footer>
            </div>
        </body>
        </html>
        """

        with open(os.path.join(templates_dir, 'report_template.html'), 'w') as f:
            f.write(report_template)