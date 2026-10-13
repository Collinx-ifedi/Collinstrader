# market_data.py
import os
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
import sys
sys.path.insert(0, '/storage/emulated/0/Android/data/ru.iiec.pydroid3/files/cryptsignal')

import logging
import ccxt
import pandas as pd
import time
import asyncio
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from utils.coin_symbol_mapper import get_symbol_from_name

logger = logging.getLogger("MarketData")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Define supported exchanges and their classes
EXCHANGE_CLASSES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "okx": ccxt.okx,
    "bitget": ccxt.bitget,
}

# Fallback exchanges in order of preference
FALLBACK_EXCHANGES = ["binance", "okx", "bitget"]

class MarketDataFetcher:
    """
    Fetches historical OHLCV market data from various exchanges using ccxt with fallback support.
    """
    def __init__(self, exchange_name: str = "binance", timeframe: str = "1h", limit: int = 500):
        """
        Initializes the MarketDataFetcher with market loading and fallback support.

        Args:
            exchange_name: Name of the cryptocurrency exchange (e.g., "binance", "bybit").
            timeframe: Default timeframe for fetching data (e.g., "1h", "1d").
            limit: Default number of candles to fetch.
        """
        self.exchange_name = exchange_name.lower()
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = None
        self.markets_loaded = False

        if self.exchange_name not in EXCHANGE_CLASSES:
            logger.error(f"Unsupported exchange specified: {self.exchange_name}")
            raise ValueError(f"Unsupported exchange: {self.exchange_name}")

        self._initialize_exchange()

    def _initialize_exchange(self, exchange_name: Optional[str] = None) -> bool:
        """
        Initializes or reinitializes the exchange with retry logic for market loading.

        Args:
            exchange_name: Exchange to initialize (defaults to self.exchange_name).

        Returns:
            bool: True if initialization and market loading succeed, False otherwise.
        """
        use_exchange_name = exchange_name or self.exchange_name
        logger.info(f"Initializing exchange: {use_exchange_name}")
        try:
            exchange_class = EXCHANGE_CLASSES[use_exchange_name]
            self.exchange = exchange_class({
                "enableRateLimit": True,
                "timeout": 30000,
                "apiVersion": "v5" if use_exchange_name == "bybit" else None,  # Explicit for Bybit
            })

            # Exchange-specific options
            if use_exchange_name == "bybit":
                self.exchange.options["defaultType"] = "spot"  # Changed to spot for SOL/USDT
            elif use_exchange_name == "okx":
                self.exchange.options["defaultType"] = "spot"

            # Load markets with retries
            for attempt in range(3):
                try:
                    self.exchange.load_markets()
                    self.markets_loaded = True
                    logger.info(f"Markets loaded for {use_exchange_name}")
                    self.exchange_name = use_exchange_name  # Update if fallback used
                    return True
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.warning(f"Attempt {attempt + 1}/3 to load markets for {use_exchange_name} failed: {type(e).__name__} - {e}")
                    if attempt < 2:
                        sleep_time = min(30, (2 ** attempt) + np.random.rand())
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed to load markets for {use_exchange_name} after 3 attempts")
                        return False
        except Exception as e:
            logger.error(f"Failed to initialize exchange '{use_exchange_name}': {e}")
            self.exchange = None
            return False

    def __del__(self):
        """Ensure the exchange connection is closed when the object is garbage collected."""
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                self.exchange.close()
            except Exception as e:
                logger.warning(f"Error closing ccxt exchange connection for {self.exchange_name}: {e}")

    def fetch_ohlcv(self, trading_symbol: str, timeframe: Optional[str] = None,
                    since: Optional[int] = None, limit: Optional[int] = None,
                    retries: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV data for a given trading symbol with fallback to other exchanges.

        Args:
            trading_symbol: The trading pair symbol (e.g., "BTC/USDT").
            timeframe: The timeframe (e.g., "1h", "1d"). Defaults to self.timeframe.
            since: Timestamp in milliseconds to fetch data from.
            limit: Number of candles to fetch. Defaults to self.limit.
            retries: Number of times to retry the fetch operation.

        Returns:
            A pandas DataFrame with timestamp as index and columns ["open", "high", "low", "close", "volume"],
            or None if fetching fails or returns empty data.
        """
        if self.exchange is None or not self.markets_loaded:
            logger.error("Exchange is not initialized or markets not loaded. Attempting fallback...")
            for fallback_exchange in FALLBACK_EXCHANGES:
                if fallback_exchange != self.exchange_name and self._initialize_exchange(fallback_exchange):
                    logger.info(f"Fallback to {fallback_exchange} successful")
                    break
            else:
                logger.error("All fallback exchanges failed. Cannot fetch OHLCV.")
                return None

        use_timeframe = timeframe or self.timeframe
        use_limit = limit or self.limit
        use_since = since

        logger.info(f"Fetching OHLCV for {trading_symbol} on {self.exchange_name} ({use_timeframe}, limit {use_limit}, since {use_since})...")

        # Check if the symbol is supported
        if trading_symbol not in self.exchange.markets:
            logger.error(f"Trading symbol '{trading_symbol}' not supported on {self.exchange_name}.")
            return None

        # Retry logic for fetching
        for attempt in range(retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(trading_symbol, timeframe=use_timeframe, since=use_since, limit=use_limit)
                if not ohlcv:
                    logger.warning(f"OHLCV fetch for {trading_symbol} from {self.exchange_name} returned no data.")
                    return None
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed for {trading_symbol} on {self.exchange_name}: {type(e).__name__} - {e}")
                if attempt < retries - 1:
                    sleep_time = min(30, (2 ** attempt) + np.random.rand())
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {retries} attempts failed for {trading_symbol} on {self.exchange_name}")
                    # Try fallback exchanges
                    for fallback_exchange in FALLBACK_EXCHANGES:
                        if fallback_exchange != self.exchange_name and self._initialize_exchange(fallback_exchange):
                            logger.info(f"Fallback to {fallback_exchange} for OHLCV fetch")
                            return self.fetch_ohlcv(trading_symbol, timeframe, since, limit, retries)
                    logger.error("All fallback exchanges failed.")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error during fetch attempt {attempt + 1} for {trading_symbol}: {e}")
                return None

        # Create DataFrame
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if ohlcv and len(ohlcv[0]) > len(columns):
            columns.extend([f'extra_col_{i}' for i in range(len(ohlcv[0]) - len(columns))])
        df = pd.DataFrame(ohlcv, columns=columns)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df.sort_index(inplace=True)

        # Ensure numeric data types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN
        initial_rows = len(df)
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows with NaN values.")

        if df.empty:
            logger.warning(f"DataFrame is empty after cleaning for {trading_symbol}.")
            return None

        logger.info(f"Successfully processed {len(df)} OHLCV candles for {trading_symbol}.")
        return df

    async def async_fetch_ohlcv(self, trading_symbol: str, timeframe: Optional[str] = None,
                                since: Optional[int] = None, limit: Optional[int] = None,
                                retries: int = 5) -> Optional[pd.DataFrame]:
        """
        Asynchronously fetches OHLCV data by running the blocking fetch_ohlcv in a thread pool.
        """
        logger.debug(f"Initiating async fetch for {trading_symbol}...")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("No running asyncio loop found. Cannot run async_fetch_ohlcv.")
            return None

        df = await loop.run_in_executor(
            None,
            self.fetch_ohlcv,
            trading_symbol, timeframe, since, limit, retries
        )
        logger.debug(f"Async fetch completed for {trading_symbol}.")
        return df

if __name__ == "__main__":
    print("--- Market Data Fetcher CLI ---")
    print("Fetch historical OHLCV data for a crypto asset.")

    async def main_cli():
        asset_input = input("Enter coin name, symbol, or trading pair: ").strip()
        trading_symbol = None
        if '/' in asset_input:
            trading_symbol = asset_input.upper()
            logger.info(f"Input '{asset_input}' treated as trading pair.")
        else:
            mapped_symbol = get_symbol_from_name(asset_input)
            if mapped_symbol:
                trading_symbol = mapped_symbol.upper()
                logger.info(f"Input '{asset_input}' mapped to '{trading_symbol}'.")
            else:
                print(f"\nCould not map '{asset_input}' to a trading pair. Try a pair like BTC/USDT.")
                return

        if trading_symbol:
            exchange_input = input(f"Enter exchange name (default 'binance'): ").strip().lower() or "binance"
            timeframe_input = input(f"Enter timeframe (default '1h'): ").strip().lower() or "1h"
            limit_input_str = input(f"Enter limit of candles (default 500): ").strip() or "500"
            try:
                limit_input = int(limit_input_str)
                if limit_input <= 0:
                    raise ValueError("Limit must be positive.")
            except ValueError:
                logger.warning(f"Invalid limit: '{limit_input_str}'. Defaulting to 500.")
                limit_input = 500

            try:
                fetcher = MarketDataFetcher(exchange_name=exchange_input, timeframe=timeframe_input, limit=limit_input)
            except ValueError as e:
                print(f"\nError initializing fetcher: {e}. Aborting.")
                return

            print("\nFetching data...")
            df = await fetcher.async_fetch_ohlcv(trading_symbol)
            if df is None or df.empty:
                print(f"\nCould not fetch data for {trading_symbol} from {exchange_input}.")
            else:
                print(f"\nSuccessfully fetched {len(df)} candles.")
                print("Head of DataFrame:")
                print(df.head())
                print("\nTail of DataFrame:")
                print(df.tail())
                print("\nDataFrame Info:")
                df.info()

    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        logger.error(f"Unexpected error in CLI: {e}")
        print(f"\nAn unexpected error occurred: {e}")