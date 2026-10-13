# src/utils/coin_symbol_mapper.py

import requests
import json
import os
import time
import logging

CACHE_DIR = os.path.dirname(__file__)
CACHE_FILE = os.path.join(CACHE_DIR, "coin_symbol_cache.json")
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/list"
CACHE_TTL = 24 * 3600  # Cache valid for 24 hours

logger = logging.getLogger("CoinSymbolMapper")
logging.basicConfig(level=logging.INFO)


def _load_local_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
            if time.time() - data.get("_timestamp", 0) < CACHE_TTL:
                return data.get("mapping", {})
    return {}


def _save_local_cache(mapping: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)  # Ensure directory exists
    data = {
        "_timestamp": time.time(),
        "mapping": mapping
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=4)


def _fetch_coingecko_mapping() -> dict:
    try:
        response = requests.get(COINGECKO_API_URL, timeout=10)
        response.raise_for_status()
        coins = response.json()
        mapping = {coin["name"].lower(): coin["symbol"].upper() for coin in coins}
        _save_local_cache(mapping)
        logger.info("Fetched and cached coin symbol mapping from CoinGecko.")
        return mapping
    except requests.RequestException as e:
        logger.warning(f"CoinGecko API fetch failed: {e}")
        return {}


def get_symbol_from_name(coin_name: str, exchange: str = "binance") -> str:
    """
    Resolve full coin name (e.g., 'Bitcoin') to symbol used by exchange (e.g., 'BTC/USDT').
    """
    coin_name = coin_name.lower()
    mapping = _load_local_cache()

    if coin_name not in mapping:
        logger.info(f"{coin_name} not in local cache. Fetching from CoinGecko.")
        mapping = _fetch_coingecko_mapping()

    symbol = mapping.get(coin_name)
    if symbol:
        trading_pair = f"{symbol}/USDT"
        return trading_pair
    else:
        logger.error(f"Could not resolve symbol for {coin_name}")
        return None


# Example usage
if __name__ == "__main__":
    print(get_symbol_from_name("Ethereum"))  # Output: ETH/USDT
    print(get_symbol_from_name("Solana"))    # Output: SOL/USDT