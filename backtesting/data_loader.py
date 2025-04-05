# silver_bullet_bot/backtesting/data_loader.py

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
import pytz


class DataLoader:
    """
    Utility for loading and preprocessing historical market data for backtesting
    """

    def __init__(self, base_data_dir='forex_data', logger=None):
        """
        Initialize the data loader

        Parameters:
        -----------
        base_data_dir : str
            Base directory containing historical data
        logger : logging.Logger
            Logger instance
        """
        self.base_data_dir = base_data_dir
        self.logger = logger or logging.getLogger('backtest')

    def discover_data_files(self, symbol, timeframes=None):
        """
        Discover all available data files for a symbol

        Parameters:
        -----------
        symbol : str
            Symbol to find data for (e.g., "NAS100")
        timeframes : list
            List of timeframes to load (e.g., ["H1", "M15"])

        Returns:
        --------
        dict
            Dictionary mapping timeframes to lists of data file paths
        """
        data_files = {}
        year_dirs = [d for d in os.listdir(self.base_data_dir)
                     if os.path.isdir(os.path.join(self.base_data_dir, d)) and d.startswith('data_')]

        # Initialize timeframes dict
        if timeframes:
            for tf in timeframes:
                data_files[tf] = []
        else:
            timeframes = []

        # Search for relevant data files
        for year_dir in year_dirs:
            symbol_dir = os.path.join(self.base_data_dir, year_dir, symbol)

            if not os.path.exists(symbol_dir):
                continue

            # Find all data files for this symbol
            for file in os.listdir(symbol_dir):
                # Parse timeframe from filename (e.g., data_h1_nas100.json -> H1)
                if file.startswith('data_') and file.endswith('.json'):
                    # Extract timeframe from filename
                    parts = file.split('_')
                    if len(parts) >= 3:
                        tf = parts[1].upper()

                        # Convert timeframe format (e.g., h1 -> H1, m15 -> M15)
                        if tf.startswith('h'):
                            tf = f"H{tf[1:]}"
                        elif tf.startswith('m'):
                            tf = f"M{tf[1:]}"
                        elif tf.startswith('d'):
                            tf = "DAILY"

                        # Add to discovered timeframes if not already specified
                        if not timeframes or tf in timeframes:
                            if tf not in data_files:
                                data_files[tf] = []

                            file_path = os.path.join(symbol_dir, file)
                            data_files[tf].append(file_path)

        # Sort files for each timeframe
        for tf in data_files:
            data_files[tf].sort()
            self.logger.info(f"Found {len(data_files[tf])} data files for {symbol} {tf}")

        return data_files

    def load_data(self, symbol, timeframes=None, start_date=None, end_date=None):
        """
        Load historical data for symbol and timeframes

        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        timeframes : list
            List of timeframes to load
        start_date : datetime or str
            Start date for data
        end_date : datetime or str
            End date for data

        Returns:
        --------
        dict
            Dictionary mapping timeframes to pandas DataFrames
        """
        # Convert date strings to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        # Discover data files
        data_files = self.discover_data_files(symbol, timeframes)

        # Initialize result dictionary
        result = {}

        # Load data for each timeframe
        for tf, files in data_files.items():
            if not files:
                self.logger.warning(f"No data files found for {symbol} {tf}")
                continue

            # Load and concatenate data from all files
            dfs = []
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Convert to DataFrame
                    if isinstance(data, dict) and 't' in data and 'o' in data:
                        # Array-style format
                        df = pd.DataFrame({
                            'time': pd.to_datetime(data['t'], unit='s'),
                            'open': data['o'],
                            'high': data['h'],
                            'low': data['l'],
                            'close': data['c'],
                            'volume': data.get('v', [0] * len(data['t']))
                        })
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        # List of dictionaries format
                        df = pd.DataFrame(data)
                        if 'time' not in df.columns and 't' in df.columns:
                            df['time'] = pd.to_datetime(df['t'], unit='s')

                    else:
                        self.logger.error(f"Unsupported data format in {file_path}")
                        continue

                    # Ensure time column is datetime type
                    if 'time' in df.columns and not pd.api.types.is_datetime64_dtype(df['time']):
                        if df['time'].dtype == 'int64':
                            df['time'] = pd.to_datetime(df['time'], unit='s')
                        else:
                            df['time'] = pd.to_datetime(df['time'])

                    dfs.append(df)
                    self.logger.debug(f"Loaded {len(df)} rows from {file_path}")

                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
                    continue

            if not dfs:
                self.logger.warning(f"No valid data loaded for {symbol} {tf}")
                continue

            # Concatenate all dataframes
            df_combined = pd.concat(dfs, ignore_index=True)

            # Remove duplicates
            df_combined = df_combined.drop_duplicates(subset=['time'])

            # Sort by time
            df_combined = df_combined.sort_values('time')

            # Filter by date range if specified
            if start_date:
                df_combined = df_combined[df_combined['time'] >= start_date]

            if end_date:
                df_combined = df_combined[df_combined['time'] <= end_date]

            # Add derived columns for analysis
            df_combined['is_bullish'] = df_combined['close'] >= df_combined['open']

            # Store in result dictionary
            result[tf] = df_combined
            self.logger.info(f"Loaded {len(df_combined)} rows for {symbol} {tf}")

        return result

    def load_multi_timeframe_data(self, symbol, timeframes, start_date=None, end_date=None):
        """
        Load data for multiple timeframes, ensuring alignment

        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        timeframes : list
            List of timeframes to load
        start_date : datetime or str
            Start date for data
        end_date : datetime or str
            End date for data

        Returns:
        --------
        dict
            Dictionary mapping timeframes to pandas DataFrames
        """
        return self.load_data(symbol, timeframes, start_date, end_date)