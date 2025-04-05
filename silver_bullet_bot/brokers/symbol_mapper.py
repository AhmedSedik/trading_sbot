# silver_bullet_bot/brokers/symbol_mapper.py

import logging
import MetaTrader5 as mt5
from silver_bullet_bot.config import INSTRUMENTS


class SymbolMapper:
    """
    Utility class to map standardized instrument names to broker-specific symbols
    """

    def __init__(self):
        """Initialize the symbol mapper"""
        self.logger = logging.getLogger('silver_bullet')
        self.symbol_map = {}
        self._initialize_symbol_map()

    def _initialize_symbol_map(self):
        """
        Initialize the symbol map by checking available symbols from the broker
        and matching them against the aliases in the configuration
        """
        self.logger.info("Initializing symbol mapper")

        # Get all available symbols from MT5
        broker_symbols = mt5.symbols_get()
        if not broker_symbols:
            self.logger.error(f"Failed to get broker symbols. Error: {mt5.last_error()}")
            return

        # Convert broker symbols to a list of names for easier lookup
        broker_symbol_names = [symbol.name for symbol in broker_symbols]
        self.logger.debug(f"Available broker symbols: {len(broker_symbol_names)}")

        # Map each configured instrument to available broker symbols
        for instrument_name, config in INSTRUMENTS.items():
            # Skip the defaults entry
            if instrument_name == "defaults":
                continue

            # If the symbol exactly matches, use it
            if config["symbol"] in broker_symbol_names:
                self.symbol_map[instrument_name] = config["symbol"]
                self.logger.debug(f"Mapped {instrument_name} to {config['symbol']} (exact match)")
                continue

            # Otherwise, look through aliases
            if "alias" in config:
                for alias in config["alias"]:
                    if alias in broker_symbol_names:
                        self.symbol_map[instrument_name] = alias
                        self.logger.debug(f"Mapped {instrument_name} to {alias} (alias match)")
                        break

            # If no mapping found, log a warning
            if instrument_name not in self.symbol_map:
                self.logger.warning(f"No symbol mapping found for {instrument_name}")

    def get_broker_symbol(self, instrument_name):
        """
        Get the broker-specific symbol for a given instrument

        Parameters:
        -----------
        instrument_name : str
            The standardized instrument name as defined in config

        Returns:
        --------
        str or None
            The broker-specific symbol, or None if not found
        """
        if instrument_name in self.symbol_map:
            return self.symbol_map[instrument_name]

        self.logger.warning(f"Symbol mapping not found for {instrument_name}")
        return None

    def get_instrument_for_symbol(self, symbol):
        """
        Get the standardized instrument name for a broker-specific symbol

        Parameters:
        -----------
        symbol : str
            The broker-specific symbol

        Returns:
        --------
        str or None
            The standardized instrument name, or None if not found
        """
        for instrument_name, broker_symbol in self.symbol_map.items():
            if broker_symbol == symbol:
                return instrument_name

        self.logger.warning(f"Instrument mapping not found for symbol {symbol}")
        return None