# technical_analysis.py
import sys
import os
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1' # For compatibility if needed
import pandas as pd
import numpy as np
import logging
import asyncio # Keep for potential async main, though LLM call is sync
from typing import Dict, Any, Optional, List
import json
import requests
import re

# --- PATH SETUP ---
# Ensure this path is correct for your environment or remove if script is in PYTHONPATH
PROJECT_ROOT = '/storage/emulated/0/Android/data/ru.iiec.pydroid3/files/cryptsignal'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- IMPORTS ---
# Attempt to import utility modules
try:
    from src.data.market_data import MarketDataFetcher # For CLI testing
    _MarketDataFetcher_imported = True
except ImportError:
    _MarketDataFetcher_imported = False
    # Define a dummy for CLI if not found, script will still load for TechnicalAnalyzer use
    class MarketDataFetcher: pass
    logging.warning("MarketDataFetcher not found. CLI testing requiring it may fail.")

try:
    from utils.coin_symbol_mapper import get_symbol_from_name
    _get_symbol_from_name_imported = True
except ImportError:
    _get_symbol_from_name_imported = False
    logging.warning("coin_symbol_mapper not found. Symbol mapping in CLI might fail.")
    def get_symbol_from_name(name: str) -> Optional[str]: return None # type: ignore

try:
    from utils.logger import get_logger
    logger = get_logger("TechnicalAnalysis") # Initialize logger early
except ImportError:
    logging.warning("get_logger utility not found. Using basic logging for TechnicalAnalysis.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("TechnicalAnalysis_Fallback")

try:
    from utils.credentials_loader import load_credentials
    _load_credentials_imported = True
except ImportError:
    _load_credentials_imported = False
    logger.error("Could not import credentials_loader. LLM API key loading from credentials.yaml will fail.")
    def load_credentials() -> Optional[Dict[str, Any]]: return None # type: ignore

# --- LLM Configuration ---
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "meta-llama/llama-4-maverick:free" # Example, use a preferred valid model
LLM_REQUEST_TIMEOUT = 90  # Increased timeout

_llm_api_key: Optional[str] = None
_credentials_loaded_attempted: bool = False

def _get_openrouter_api_key() -> Optional[str]:
    global _llm_api_key, _credentials_loaded_attempted
    
    if _llm_api_key is not None: # Already fetched
        return _llm_api_key

    # 1. Try environment variable first
    _llm_api_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
    if _llm_api_key:
        logger.info("OpenRouter API key loaded from environment variable.")
        return _llm_api_key

    # 2. If not in environment, try loading from credentials.yaml (if loader imported)
    if not _credentials_loaded_attempted and _load_credentials_imported and callable(load_credentials):
        try:
            credentials = load_credentials() # Load the credentials file
            if credentials:
                _llm_api_key = credentials.get('openrouter', {}).get('api_key')
                if _llm_api_key:
                    logger.info("OpenRouter API key loaded from credentials.yaml.")
                else:
                    logger.warning("OpenRouter API key not found in credentials.yaml under 'openrouter.api_key'.")
            else:
                logger.warning("Failed to load credentials from file (credentials.yaml might be empty or misformatted).")
        except FileNotFoundError:
            logger.warning("credentials.yaml not found. Skipping API key load from file.")
        except Exception as e:
            logger.error(f"Error loading credentials from file: {e}", exc_info=True)
        _credentials_loaded_attempted = True # Mark as attempted to load from file

    if _llm_api_key is None:
        logger.error(f"{OPENROUTER_API_KEY_ENV_VAR} not found in environment or credentials.yaml. LLM technical analysis will be skipped.")
    return _llm_api_key

def _extract_json_from_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        logger.warning("LLM response text is empty, cannot extract JSON.")
        return None
    # More robust regex to find JSON block, including if it's not perfectly formatted with ```json
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})(?=\s*$)', response_text, re.DOTALL)
    if match:
        json_str = match.group(1) if match.group(1) else match.group(2)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from LLM TA response: {e}. JSON String: '{json_str}'")
                return None
        else:
            logger.warning("Regex matched but captured JSON string was empty.") # Should be rare
            return None
            
    logger.warning(f"No distinct JSON object found in LLM TA response. Response snippet: '{response_text[:300]}...'")
    return None

def _call_openrouter_llm_for_ta(narrative_summary: str, symbol: str, timeframe: str, api_key: str) -> Dict[str, Any]:
    default_error_response = {
        "summary_text": "LLM technical analysis summary is unavailable.",
        "overall_sentiment_category": "neutral",
        "numeric_sentiment": 0,
        "error": "LLM call failed or API key missing.",
        "indicator_narrative_fallback": narrative_summary # Include the prepared narrative
    }
    if not api_key:
        logger.error("OpenRouter API key not configured. Cannot call LLM for TA.")
        return {**default_error_response, "error": "API key not configured."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Replace with your actual site if deployed
        "X-Title": "CryptoSignal Technical Analysis Module"
    }
    system_prompt = (
        "You are an expert technical analyst AI. Your task is to provide a concise, holistic analysis "
        "of a cryptocurrency's market sentiment based on a summary of its technical indicators. "
        "Focus on the combined implications."
    )
    user_prompt = (
        f"Asset: {symbol}\nTimeframe: {timeframe}\n\n"
        f"Indicator Summary:\n{narrative_summary}\n\n"
        "Instructions:\n"
        "1. Provide a brief (2-4 sentences) holistic summary of the combined implications of these indicators. "
        "Identify key confluences or divergences.\n"
        "2. Conclude with an overall market sentiment classification: 'bullish', 'neutral', or 'bearish'.\n\n"
        "Return your response ONLY as a single well-formed JSON object with three keys:\n"
        "- \"summary_text\" (string): Your holistic analysis summary.\n"
        "- \"overall_sentiment_category\" (string): Must be one of 'bullish', 'neutral', or 'bearish'.\n"
        "- \"numeric_sentiment\" (integer): 1 for bullish, 0 for neutral, -1 for bearish.\n\n"
        "Example JSON output:\n"
        "{\n"
        "  \"summary_text\": \"The indicators suggest strong bullish momentum with price above key moving averages and a positive MACD, though RSI is approaching overbought levels, indicating potential for a short-term pullback within the uptrend.\",\n"
        "  \"overall_sentiment_category\": \"bullish\",\n"
        "  \"numeric_sentiment\": 1\n"
        "}"
    )
    payload = {
        "model": DEFAULT_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 350, # Increased slightly for potentially more nuanced summary
        "temperature": 0.3, # Low temperature for factual, less creative analysis
        "response_format": {"type": "json_object"} # Request JSON directly if model supports
    }

    try:
        logger.debug(f"Sending TA request to LLM for {symbol} ({timeframe}). Narrative snippet: {narrative_summary[:100]}...")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        
        llm_response_content = response.text # Get raw text first
        logger.debug(f"Raw LLM TA response for {symbol}: {llm_response_content}")
        
        # Attempt to parse directly if response_format worked, else use existing _extract_json
        try:
            parsed_json = json.loads(llm_response_content)
            logger.info("Successfully parsed LLM JSON response directly.")
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as direct JSON, trying regex extraction.")
            parsed_json = _extract_json_from_llm_response(llm_response_content)

        if parsed_json and \
           isinstance(parsed_json.get("summary_text"), str) and \
           isinstance(parsed_json.get("overall_sentiment_category"), str) and \
           isinstance(parsed_json.get("numeric_sentiment"), int):
            
            category = parsed_json["overall_sentiment_category"].lower()
            num_sentiment = parsed_json["numeric_sentiment"]
            
            # Validate and reconcile category and numeric sentiment
            valid_categories = ['bullish', 'neutral', 'bearish']
            valid_numerics = { 'bullish': 1, 'neutral': 0, 'bearish': -1 }

            if category not in valid_categories:
                logger.warning(f"LLM returned invalid sentiment category for {symbol}: '{category}'. Attempting to infer or default.")
                # Attempt to infer from numeric_sentiment if category is bad
                if num_sentiment == 1: category = 'bullish'
                elif num_sentiment == -1: category = 'bearish'
                else: category = 'neutral' # Default if numeric also unclear
            
            # Reconcile numeric_sentiment with (potentially corrected) category
            if num_sentiment != valid_numerics.get(category):
                logger.warning(f"LLM returned inconsistent numeric sentiment for {symbol}: Numeric was {num_sentiment} for category '{category}'. Reconciling.")
                num_sentiment = valid_numerics.get(category, 0) # Default to 0 if category somehow still invalid

            parsed_json["overall_sentiment_category"] = category
            parsed_json["numeric_sentiment"] = num_sentiment
            
            return {**parsed_json, "error": None, "indicator_narrative_fallback": narrative_summary}
        
        logger.error(f"LLM TA response parsing or validation failed for {symbol}. Parsed JSON: {parsed_json if 'parsed_json' in locals() else 'Not parsed'}. Raw response: {llm_response_content[:300]}")
        return {**default_error_response, "error": "LLM response parsing or validation failed."}

    except requests.exceptions.Timeout:
        logger.error(f"LLM TA request timed out for {symbol}.")
        return {**default_error_response, "error": "LLM request timed out."}
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text[:200] if e.response else str(e)
        status_code = e.response.status_code if e.response is not None else "N/A"
        logger.error(f"LLM TA HTTP error for {symbol} (Status {status_code}): {error_text}")
        return {**default_error_response, "error": f"LLM API HTTP error {status_code}."}
    except Exception as e:
        logger.error(f"Unexpected error during LLM TA call for {symbol}: {e}", exc_info=True)
        return {**default_error_response, "error": f"Unexpected error: {str(e)}."}


class TechnicalAnalyzer:
    def __init__(self, df_ohlcv: pd.DataFrame):
        if not isinstance(df_ohlcv, pd.DataFrame):
            logger.error("TechnicalAnalyzer initialized with non-DataFrame input.")
            self.df_ohlcv_original = pd.DataFrame() # Store original
            self.df_with_indicators = pd.DataFrame()
            return

        self.df_ohlcv_original = df_ohlcv.copy() # Keep a pristine copy of original OHLCV
        
        if df_ohlcv.empty:
            logger.warning("TechnicalAnalyzer initialized with empty DataFrame.")
            self.df_with_indicators = pd.DataFrame()
            return

        # Work on a new copy for modifications and indicators
        df_processed = df_ohlcv.copy()

        # Standardize column names
        rename_map = {
            "Timestamp": "timestamp", "Date": "timestamp", "date": "timestamp",
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        }
        # Filter rename_map to only include columns present in df_processed
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df_processed.columns}
        df_processed.rename(columns=actual_rename_map, inplace=True)
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        # Timestamp handling
        if 'timestamp' in df_processed.columns:
            try:
                df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
                if not pd.api.types.is_datetime64_any_dtype(df_processed.index): # Set index if not already datetime
                    df_processed.set_index('timestamp', inplace=True, drop=False) # Keep timestamp column too
            except Exception as e:
                logger.error(f"Error converting 'timestamp' column to datetime: {e}. Proceeding without proper time index if conversion failed.")
        elif pd.api.types.is_datetime64_any_dtype(df_processed.index): # If index is already datetime
             df_processed['timestamp'] = df_processed.index # Ensure timestamp column exists from index
        else:
            logger.warning("DataFrame has no 'timestamp' column and index is not datetime. Time-based operations might fail.")


        # Convert OHLCV columns to numeric, coercing errors
        for col in required_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            else:
                logger.warning(f"Required column '{col}' not found in DataFrame after renaming. Indicator generation may fail.")
                # Create an empty column to prevent key errors later, though results will be poor
                df_processed[col] = np.nan


        # Drop rows where essential OHLCV data is NaN AFTER conversion
        # This is important before calculating indicators from numeric inputs.
        initial_rows = len(df_processed)
        df_processed.dropna(subset=required_cols, inplace=True)
        if len(df_processed) < initial_rows:
            logger.info(f"Dropped {initial_rows - len(df_processed)} rows with NaN in essential OHLCV columns.")

        self.df_with_indicators = df_processed # This DataFrame will hold OHLCV + indicators

        # Indicator calculations are implemented below using pandas and numpy only.
        logger.info("TechnicalAnalyzer initialized with built-in pandas/numpy indicators.")

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the module's complete indicator set from OHLCV using pandas/numpy."""
        out = df.copy()
        high, low, close, volume = out["high"], out["low"], out["close"], out["volume"]
        open_ = out["open"]

        # Moving averages and trend baselines
        for length in (20, 50, 200):
            out[f"SMA_{length}"] = close.rolling(length, min_periods=length).mean()
        for length in (12, 20, 26, 50, 200):
            out[f"EMA_{length}"] = close.ewm(span=length, adjust=False, min_periods=length).mean()

        # MACD
        ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        out["MACD_12_26_9"] = macd
        out["MACDs_12_26_9"] = macd_signal
        out["MACDh_12_26_9"] = macd - macd_signal

        # Ichimoku Cloud (Chikou is the close shifted backwards in conventional charts)
        tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2
        kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2
        span_b = (high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2
        out["ITS_9"] = tenkan
        out["IKS_26"] = kijun
        out["ISA_9"] = ((tenkan + kijun) / 2).shift(26)
        out["ISB_26"] = span_b.shift(26)
        out["ICS_26"] = close.shift(-26)

        # True range and Wilder-smoothed ATR
        previous_close = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1
        ).max(axis=1)
        out["TR"] = true_range
        atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        out["ATRr_14"] = atr

        # Directional movement and ADX
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
        smoothed_tr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        plus_di = 100 * self._safe_divide(plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(), smoothed_tr)
        minus_di = 100 * self._safe_divide(minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(), smoothed_tr)
        dx = 100 * self._safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
        out["DMP_14"] = plus_di
        out["DMN_14"] = minus_di
        out["ADX_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        # Aroon oscillator
        def aroon_up(window: pd.Series) -> float:
            return 100.0 * (window.size - 1 - int(np.argmax(window.to_numpy()))) / (window.size - 1) if window.size > 1 else np.nan
        def aroon_down(window: pd.Series) -> float:
            return 100.0 * (window.size - 1 - int(np.argmin(window.to_numpy()))) / (window.size - 1) if window.size > 1 else np.nan
        out["AROONU_14"] = high.rolling(15, min_periods=15).apply(aroon_up, raw=False)
        out["AROOND_14"] = low.rolling(15, min_periods=15).apply(aroon_down, raw=False)
        out["AROONOSC_14"] = out["AROONU_14"] - out["AROOND_14"]

        # Parabolic SAR (iterative implementation)
        psar_long = pd.Series(np.nan, index=out.index, dtype=float)
        psar_short = pd.Series(np.nan, index=out.index, dtype=float)
        if len(out) >= 2:
            is_up = close.iloc[1] >= close.iloc[0]
            sar = float(low.iloc[0] if is_up else high.iloc[0])
            extreme = float(high.iloc[0] if is_up else low.iloc[0])
            acceleration = 0.02
            for i in range(1, len(out)):
                prior_sar = sar
                sar = prior_sar + acceleration * (extreme - prior_sar)
                if is_up:
                    sar = min(sar, float(low.iloc[i - 1]))
                    if i >= 2:
                        sar = min(sar, float(low.iloc[i - 2]))
                    if low.iloc[i] < sar:
                        is_up, sar, extreme, acceleration = False, extreme, float(low.iloc[i]), 0.02
                    else:
                        if high.iloc[i] > extreme:
                            extreme = float(high.iloc[i])
                            acceleration = min(acceleration + 0.02, 0.2)
                else:
                    sar = max(sar, float(high.iloc[i - 1]))
                    if i >= 2:
                        sar = max(sar, float(high.iloc[i - 2]))
                    if high.iloc[i] > sar:
                        is_up, sar, extreme, acceleration = True, extreme, float(high.iloc[i]), 0.02
                    else:
                        if low.iloc[i] < extreme:
                            extreme = float(low.iloc[i])
                            acceleration = min(acceleration + 0.02, 0.2)
                if is_up:
                    psar_long.iloc[i] = sar
                else:
                    psar_short.iloc[i] = sar
        out["PSARl_0.02_0.2"] = psar_long
        out["PSARs_0.02_0.2"] = psar_short
        out["PSARaf_0.02_0.2"] = np.nan
        out["PSARr_0.02_0.2"] = np.nan

        # RSI (Wilder smoothing)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = self._safe_divide(avg_gain, avg_loss)
        out["RSI_14"] = 100 - (100 / (1 + rs))
        out.loc[(avg_loss == 0) & (avg_gain > 0), "RSI_14"] = 100
        out.loc[(avg_loss == 0) & (avg_gain == 0), "RSI_14"] = 50

        # Stochastic oscillator and Stochastic RSI
        lowest14 = low.rolling(14, min_periods=14).min()
        highest14 = high.rolling(14, min_periods=14).max()
        stoch_raw = 100 * self._safe_divide(close - lowest14, highest14 - lowest14)
        out["STOCHk_14_3_3"] = stoch_raw.rolling(3, min_periods=3).mean()
        out["STOCHd_14_3_3"] = out["STOCHk_14_3_3"].rolling(3, min_periods=3).mean()
        rsi_min = out["RSI_14"].rolling(14, min_periods=14).min()
        rsi_max = out["RSI_14"].rolling(14, min_periods=14).max()
        stoch_rsi_raw = 100 * self._safe_divide(out["RSI_14"] - rsi_min, rsi_max - rsi_min)
        out["STOCHRSIk_14_14_3_3"] = stoch_rsi_raw.rolling(3, min_periods=3).mean()
        out["STOCHRSId_14_14_3_3"] = out["STOCHRSIk_14_14_3_3"].rolling(3, min_periods=3).mean()

        # CCI, ROC, momentum, PPO, slopes, Williams %R
        typical = (high + low + close) / 3
        typical_sma = typical.rolling(20, min_periods=20).mean()
        mean_deviation = typical.rolling(20, min_periods=20).apply(
            lambda x: float(np.mean(np.abs(x - np.mean(x))),), raw=True
        )
        out["CCI_20"] = self._safe_divide(typical - typical_sma, 0.015 * mean_deviation)
        out["ROC_12"] = 100 * (self._safe_divide(close, close.shift(12)) - 1)
        out["MOM_10"] = close - close.shift(10)
        out["MOM20_20"] = close - close.shift(20)
        ema12_ppo = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26_ppo = close.ewm(span=26, adjust=False, min_periods=26).mean()
        ppo = 100 * self._safe_divide(ema12_ppo - ema26_ppo, ema26_ppo)
        ppo_signal = ppo.ewm(span=9, adjust=False, min_periods=9).mean()
        out["PPO_12_26_9"] = ppo
        out["PPOh_12_26_9"] = ppo - ppo_signal
        out["PPOs_12_26_9"] = ppo_signal
        def rolling_linear_slope(values: np.ndarray) -> float:
            if len(values) < 2 or not np.isfinite(values).all():
                return np.nan
            x = np.arange(len(values), dtype=float)
            return float(np.polyfit(x, values, 1)[0])
        out["SLOPE_14"] = close.rolling(14, min_periods=14).apply(rolling_linear_slope, raw=True)
        out["SLOPE30_30"] = close.rolling(30, min_periods=30).apply(rolling_linear_slope, raw=True)
        out["WILLR_14"] = -100 * self._safe_divide(highest14 - close, highest14 - lowest14)

        # Bollinger Bands
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std(ddof=0)
        bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
        out["BBL_20_2.0"], out["BBM_20_2.0"], out["BBU_20_2.0"] = bb_lower, bb_mid, bb_upper
        out["BBB_20_2.0"] = 100 * self._safe_divide(bb_upper - bb_lower, bb_mid)
        out["BBP_20_2.0"] = self._safe_divide(close - bb_lower, bb_upper - bb_lower)

        # Donchian and Keltner channels
        dc_low = low.rolling(20, min_periods=20).min()
        dc_high = high.rolling(20, min_periods=20).max()
        out["DCL_20_20"] = dc_low
        out["DCM_20_20"] = (dc_low + dc_high) / 2
        out["DCU_20_20"] = dc_high
        kc_mid = close.ewm(span=20, adjust=False, min_periods=20).mean()
        out["KCL_20_2"] = kc_mid - 2 * atr
        out["KCB_20_2"] = kc_mid
        out["KCU_20_2"] = kc_mid + 2 * atr

        # Choppiness index and TTM squeeze state
        tr_sum = true_range.rolling(14, min_periods=14).sum()
        chop_range = high.rolling(14, min_periods=14).max() - low.rolling(14, min_periods=14).min()
        out["CHOP_14_1_100"] = 100 * np.log10(self._safe_divide(tr_sum, chop_range)) / np.log10(14)
        out["SQZ_ON"] = ((bb_lower > out["KCL_20_2"]) & (bb_upper < out["KCU_20_2"])).astype(float)
        out.loc[bb_lower.isna() | out["KCL_20_2"].isna(), "SQZ_ON"] = np.nan

        # Schaff Trend Cycle approximation: stochastic transforms of MACD, smoothed twice.
        stc_low = macd.rolling(23, min_periods=23).min()
        stc_high = macd.rolling(23, min_periods=23).max()
        stc_k = 100 * self._safe_divide(macd - stc_low, stc_high - stc_low)
        stc_d = stc_k.ewm(span=3, adjust=False, min_periods=3).mean()
        stc2_low = stc_d.rolling(23, min_periods=23).min()
        stc2_high = stc_d.rolling(23, min_periods=23).max()
        stc = 100 * self._safe_divide(stc_d - stc2_low, stc2_high - stc2_low)
        out["STC_10_12_23"] = stc.ewm(span=3, adjust=False, min_periods=3).mean()

        # Vertical Horizontal Filter
        vhf_range = close.rolling(28, min_periods=28).max() - close.rolling(28, min_periods=28).min()
        out["VHF_28"] = self._safe_divide(vhf_range, close.diff().abs().rolling(27, min_periods=27).sum())

        # Volume indicators: OBV, CMF, MFI, Elder Force Index
        direction = np.sign(close.diff()).fillna(0)
        out["OBV"] = (direction * volume).cumsum()
        money_flow_multiplier = self._safe_divide((close - low) - (high - close), high - low).fillna(0)
        money_flow_volume = money_flow_multiplier * volume
        out["CMF_20"] = self._safe_divide(
            money_flow_volume.rolling(20, min_periods=20).sum(),
            volume.rolling(20, min_periods=20).sum()
        )
        raw_money_flow = typical * volume
        positive_flow = raw_money_flow.where(typical.diff() > 0, 0.0)
        negative_flow = raw_money_flow.where(typical.diff() < 0, 0.0)
        positive_sum = positive_flow.rolling(14, min_periods=14).sum()
        negative_sum = negative_flow.rolling(14, min_periods=14).sum()
        money_ratio = self._safe_divide(positive_sum, negative_sum)
        out["MFI_14"] = 100 - (100 / (1 + money_ratio))
        out.loc[(negative_sum == 0) & (positive_sum > 0), "MFI_14"] = 100
        out["EFI_13"] = (close.diff() * volume).ewm(span=13, adjust=False, min_periods=13).mean()

        # VWAP resets each calendar day when a datetime index is available.
        if isinstance(out.index, pd.DatetimeIndex):
            typical_volume = typical * volume
            day_groups = pd.Series(out.index.date, index=out.index)
            cumulative_pv = typical_volume.groupby(day_groups).cumsum()
            cumulative_volume = volume.groupby(day_groups).cumsum()
            out["VWAP_D"] = self._safe_divide(cumulative_pv, cumulative_volume)
        else:
            out["VWAP_D"] = self._safe_divide((typical * volume).cumsum(), volume.cumsum())

        # Rolling volume profile approximation: rolling VWAP-like POC and value area.
        # OHLCV bars do not reveal true traded-volume-at-price distributions.
        out["VP_POC"] = self._safe_divide((typical * volume).rolling(10, min_periods=10).sum(),
                                           volume.rolling(10, min_periods=10).sum())
        rolling_low = low.rolling(10, min_periods=10).min()
        rolling_high = high.rolling(10, min_periods=10).max()
        out["VP_VAL"] = rolling_low
        out["VP_VAH"] = rolling_high

        return out

    def _add_golden_death_crosses(self) -> None:
        # Operates on self.df_with_indicators
        if 'SMA_50' in self.df_with_indicators.columns and 'SMA_200' in self.df_with_indicators.columns:
            # Golden Cross: SMA50 crosses above SMA200
            condition_golden_curr = (self.df_with_indicators['SMA_50'] > self.df_with_indicators['SMA_200'])
            condition_golden_prev = (self.df_with_indicators['SMA_50'].shift(1) <= self.df_with_indicators['SMA_200'].shift(1))
            self.df_with_indicators['GOLDEN_X'] = (condition_golden_curr & condition_golden_prev).astype(int)

            # Death Cross: SMA50 crosses below SMA200
            condition_death_curr = (self.df_with_indicators['SMA_50'] < self.df_with_indicators['SMA_200'])
            condition_death_prev = (self.df_with_indicators['SMA_50'].shift(1) >= self.df_with_indicators['SMA_200'].shift(1))
            self.df_with_indicators['DEATH_X'] = (condition_death_curr & condition_death_prev).astype(int)
        else:
            logger.warning("SMA_50 or SMA_200 not found in indicator DataFrame. Cannot calculate Golden/Death crosses.")
            self.df_with_indicators['GOLDEN_X'] = 0
            self.df_with_indicators['DEATH_X'] = 0


    def generate_all_indicators(self) -> Optional[pd.DataFrame]:
        if self.df_with_indicators.empty or not all(
            col in self.df_with_indicators.columns for col in ["open", "high", "low", "close", "volume"]
        ):
            logger.error("OHLCV DataFrame is empty or missing required columns for indicator generation.")
            return None

        if len(self.df_with_indicators) < 200:
            logger.warning(
                f"Insufficient data ({len(self.df_with_indicators)} rows) for some long-period indicators (e.g., SMA200). Results may be sparse."
            )

        try:
            self.df_with_indicators = self._calculate_indicators(self.df_with_indicators)
            self._add_golden_death_crosses()
            num_generated_cols = len(self.df_with_indicators.columns) - len(self.df_ohlcv_original.columns)
            logger.info(f"Successfully generated built-in technical indicators. Added {num_generated_cols} columns.")
            return self.df_with_indicators.copy()
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}", exc_info=True)
            return self.df_with_indicators.copy() if not self.df_with_indicators.empty else None


    def _prepare_indicator_narrative(self, symbol: str, timeframe: str) -> str:
        # Uses self.df_with_indicators which should have indicators calculated
        if self.df_with_indicators.empty or len(self.df_with_indicators) == 0:
            logger.warning("Indicator DataFrame is empty. Cannot prepare narrative.")
            return "Indicator data is not available due to empty DataFrame."
        
        latest = self.df_with_indicators.iloc[-1]
        if latest.isnull().all(): # Check if the latest row is all NaNs
            logger.warning("Latest row of indicators is all NaNs. Narrative will be sparse.")
            return "Latest indicator data is all NaNs. Cannot prepare narrative."

        narrative_parts = [f"Technical Analysis Summary for {symbol} (Timeframe: {timeframe}):"]
        close_price = latest.get('close', np.nan) # Use np.nan as default for easier checking

        # RSI
        rsi_val = latest.get('RSI_14')
        if pd.notna(rsi_val):
            rsi_interp = "neutral"
            if rsi_val > 70: rsi_interp = "overbought (bearish reversal risk)"
            elif rsi_val < 30: rsi_interp = "oversold (bullish reversal potential)"
            elif rsi_val > 55: rsi_interp = "strong bullish momentum" # Adjusted threshold
            elif rsi_val < 45: rsi_interp = "strong bearish momentum" # Adjusted threshold
            elif rsi_val > 50: rsi_interp = "leaning bullish"
            elif rsi_val < 50: rsi_interp = "leaning bearish"
            narrative_parts.append(f"- RSI(14): {rsi_val:.2f} ({rsi_interp}).")

        # MACD
        macd_line = latest.get('MACD_12_26_9')
        macd_signal = latest.get('MACDs_12_26_9')
        macd_hist = latest.get('MACDh_12_26_9')
        if pd.notna(macd_line) and pd.notna(macd_signal) and pd.notna(macd_hist):
            macd_interp = "neutral"
            if macd_line > macd_signal and macd_hist > 0: macd_interp = "bullish crossover / upward momentum"
            elif macd_line < macd_signal and macd_hist < 0: macd_interp = "bearish crossover / downward momentum"
            elif macd_line > macd_signal and macd_hist < 0: macd_interp = "potential bearish divergence or weakening bullish"
            elif macd_line < macd_signal and macd_hist > 0: macd_interp = "potential bullish divergence or weakening bearish"
            narrative_parts.append(f"- MACD(12,26,9): Line={macd_line:.2f}, Signal={macd_signal:.2f}, Hist={macd_hist:.2f} ({macd_interp}).")

        # Moving Averages & Price (ensure price is valid)
        if pd.notna(close_price):
            for ma_len in [20, 50, 200]:
                sma_col = f'SMA_{ma_len}'
                ema_col = f'EMA_{ma_len}'
                ma_val_sma = latest.get(sma_col)
                ma_val_ema = latest.get(ema_col)

                if pd.notna(ma_val_sma):
                    relation_sma = 'Above (bullish)' if close_price > ma_val_sma else 'Below (bearish)'
                    narrative_parts.append(f"- Price ({close_price:.2f}) vs {sma_col} ({ma_val_sma:.2f}): {relation_sma}.")
                if pd.notna(ma_val_ema) and ma_len != 200: # EMA200 often reported with SMA200
                    relation_ema = 'Above (bullish)' if close_price > ma_val_ema else 'Below (bearish)'
                    narrative_parts.append(f"- Price ({close_price:.2f}) vs {ema_col} ({ma_val_ema:.2f}): {relation_ema}.")
        else:
            narrative_parts.append("- Close price is NaN, cannot compare with MAs.")


        # Golden/Death Crosses
        if latest.get('GOLDEN_X') == 1: narrative_parts.append("- Golden Cross (SMA50/200) detected recently (bullish).")
        if latest.get('DEATH_X') == 1: narrative_parts.append("- Death Cross (SMA50/200) detected recently (bearish).")

        # Bollinger Bands
        bbp = latest.get('BBP_20_2.0') # Percent B
        bbu = latest.get('BBU_20_2.0') # Upper Band
        bbl = latest.get('BBL_20_2.0') # Lower Band
        bbm = latest.get('BBM_20_2.0') # Middle band (SMA20)
        if pd.notna(bbp) and pd.notna(bbu) and pd.notna(bbm) and pd.notna(bbl):
            bb_interp = "within bands"
            if bbp > 1.0: bb_interp = "above upper band (strong bullish momentum, potential overextension or breakout)"
            elif bbp < 0.0: bb_interp = "below lower band (strong bearish momentum, potential oversold or breakdown)"
            elif bbp > 0.8: bb_interp = "near upper band (testing resistance)"
            elif bbp < 0.2: bb_interp = "near lower band (testing support)"
            narrative_parts.append(f"- Bollinger Bands(20,2): Price at {bbp:.2%} of band width (L:{bbl:.2f}, M:{bbm:.2f}, U:{bbu:.2f}). Interpretation: {bb_interp}.")

        # ADX
        adx_val = latest.get('ADX_14')
        dmp_val = latest.get('DMP_14') # +DI
        dmn_val = latest.get('DMN_14') # -DI
        if pd.notna(adx_val) and pd.notna(dmp_val) and pd.notna(dmn_val):
            adx_interp = "weak or no trend"
            trend_direction = "neutral"
            if adx_val > 25:
                adx_interp = "strong trend"
                if dmp_val > dmn_val: trend_direction = "uptrend"
                else: trend_direction = "downtrend"
            elif adx_val > 20:
                adx_interp = "emerging trend"
                if dmp_val > dmn_val: trend_direction = "emerging uptrend"
                else: trend_direction = "emerging downtrend"
            narrative_parts.append(f"- ADX(14): {adx_val:.2f} ({adx_interp}). Trend: {trend_direction} (+DI:{dmp_val:.2f}, -DI:{dmn_val:.2f}).")

        # Stochastic %K
        stoch_k = latest.get('STOCHk_14_3_3')
        if pd.notna(stoch_k):
            stoch_interp = "neutral"
            if stoch_k > 80: stoch_interp = "overbought zone"
            elif stoch_k < 20: stoch_interp = "oversold zone"
            narrative_parts.append(f"- Stochastic(14,3,3) %K: {stoch_k:.2f} ({stoch_interp}).")
        
        # Volume indicators (OBV trend, MFI)
        if len(self.df_with_indicators) > 1:
            obv_val = latest.get('OBV')
            prev_obv = self.df_with_indicators['OBV'].iloc[-2] if 'OBV' in self.df_with_indicators.columns else np.nan
            if pd.notna(obv_val) and pd.notna(prev_obv):
                obv_trend = "flat"
                if obv_val > prev_obv: obv_trend = "rising (bullish volume confirmation)"
                elif obv_val < prev_obv: obv_trend = "falling (bearish volume confirmation)"
                narrative_parts.append(f"- OBV Trend: Latest value suggests volume is {obv_trend}.")

        mfi_val = latest.get('MFI_14')
        if pd.notna(mfi_val):
            mfi_interp = "neutral"
            if mfi_val > 80: mfi_interp = "overbought (potential distribution)"
            elif mfi_val < 20: mfi_interp = "oversold (potential accumulation)"
            narrative_parts.append(f"- MFI(14): {mfi_val:.2f} ({mfi_interp}).")
            
        # TTM Squeeze
        sqz_on = latest.get('SQZ_ON') # Built-in squeeze state: 1 means volatility compression is active
        if pd.notna(sqz_on):
            if sqz_on == 1: narrative_parts.append("- TTM Squeeze: ON (consolidation, potential for a strong move).")
            else: narrative_parts.append("- TTM Squeeze: OFF (market in a directional move).")


        return "\n".join(narrative_parts)


    def get_llm_overall_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generates a holistic technical analysis summary using an LLM.
        If LLM fails, includes the prepared indicator narrative as a fallback.
        Returns: Dict with summary_text, overall_sentiment_category, numeric_sentiment, error, and indicator_narrative_fallback.
        """
        api_key = _get_openrouter_api_key()
        
        # Prepare narrative regardless of API key, as it's the fallback
        if self.df_with_indicators.empty or len(self.df_with_indicators) == 0:
            # Try to generate indicators if not already done or if df is empty
            logger.info("Indicator DataFrame is empty in get_llm_overall_analysis, attempting to generate.")
            if self.generate_all_indicators() is None or self.df_with_indicators.empty:
                logger.error(f"Failed to generate indicators for {symbol} for LLM analysis.")
                return {
                    "summary_text": "Analysis unavailable: Indicator generation failed.",
                    "overall_sentiment_category": "neutral",
                    "numeric_sentiment": 0,
                    "error": "Indicator generation failed prior to LLM call.",
                    "indicator_narrative_fallback": "Indicator data could not be generated."
                }
        
        narrative_summary = self._prepare_indicator_narrative(symbol, timeframe)

        default_response_structure = {
            "summary_text": "LLM analysis could not be performed.",
            "overall_sentiment_category": "neutral",
            "numeric_sentiment": 0,
            "error": "LLM analysis skipped or failed.",
            "indicator_narrative_fallback": narrative_summary # Always include the narrative
        }

        if not api_key:
            logger.error("OpenRouter API Key not available for LLM TA. LLM analysis skipped.")
            return {**default_response_structure, "error": "API key not set."}

        if narrative_summary.startswith("Indicator data is not available"):
             logger.error("Cannot call LLM: Indicator narrative preparation failed.")
             return {**default_response_structure, "error": narrative_summary}

        # Call the LLM
        llm_result = _call_openrouter_llm_for_ta(narrative_summary, symbol, timeframe, api_key)
        
        # Ensure the fallback narrative is part of the result, even if LLM call was attempted
        # _call_openrouter_llm_for_ta already includes it in its default_error_response and success response
        return llm_result


# --- Main CLI for testing ---
async def main_cli():
    print("--- Technical Analysis Module CLI Test ---")
    
    if not _MarketDataFetcher_imported:
        print("ERROR: MarketDataFetcher is not available. Please check imports for CLI testing.")
        logger.critical("MarketDataFetcher not loaded, CLI testing cannot function properly.")
        return

    # md_fetcher instance moved into the loop for dynamic exchange selection
    coin_name_input = input("Enter cryptocurrency name (e.g., Bitcoin, Ethereum) or pair (BTC/USDT): ").strip()
    if not coin_name_input:
        print("No coin name entered. Exiting.")
        return

    trading_symbol = None
    if '/' in coin_name_input: # User likely entered a trading pair
        trading_symbol = coin_name_input.upper()
        logger.info(f"Input '{coin_name_input}' treated as a direct trading pair: {trading_symbol}")
    elif _get_symbol_from_name_imported and callable(get_symbol_from_name):
        trading_symbol = get_symbol_from_name(coin_name_input) # Defaults to /USDT pair
        if trading_symbol:
            logger.info(f"Mapped '{coin_name_input}' to trading symbol: {trading_symbol}")
        else:
            logger.warning(f"Could not map '{coin_name_input}' using coin_symbol_mapper.")
    
    if not trading_symbol: # Fallback if mapping failed or was not attempted
        default_pair = f"{coin_name_input.upper()}/USDT"
        trading_symbol_input_direct = input(f"Could not auto-map '{coin_name_input}'. Enter trading symbol (e.g., BTC/USDT) [Default: {default_pair}]: ").strip().upper()
        if trading_symbol_input_direct:
            trading_symbol = trading_symbol_input_direct
        elif default_pair:
             trading_symbol = default_pair
             logger.info(f"Using default pair {trading_symbol} for input '{coin_name_input}'.")
        else: # Should not happen if coin_name_input was provided
            print("No trading symbol could be determined. Exiting.")
            return


    exchange_input = input("Enter exchange ID (e.g., binance, kraken, coinbase) [Default: binance]: ").strip().lower() or "binance"
    timeframe_input = input("Enter timeframe (e.g., 1h, 4h, 1d) [Default: 1d]: ").strip() or "1d"
    
    limit_input_str = input("Enter number of candles to fetch [Default: 300]: ").strip() or "300"
    try:
        limit_input = int(limit_input_str)
        if limit_input <= 0: raise ValueError("Limit must be positive.")
    except ValueError:
        logger.warning(f"Invalid limit '{limit_input_str}'. Defaulting to 300.")
        limit_input = 300


    df_ohlcv = None
    try:
        logger.info(f"Fetching OHLCV for {trading_symbol} from {exchange_input} (Timeframe: {timeframe_input}, Limit: {limit_input})")
        
        # Initialize MarketDataFetcher with the specified exchange_input
        # Ensure MarketDataFetcher is the actual class, not the dummy
        if not hasattr(MarketDataFetcher, 'async_fetch_ohlcv'): # Check if it's the dummy
             print("ERROR: MarketDataFetcher is not properly initialized for fetching data (might be a dummy class).")
             return

        md_fetcher = MarketDataFetcher(exchange_name=exchange_input) # MarketDataFetcher sets its own default timeframe/limit
        
        df_ohlcv = await md_fetcher.async_fetch_ohlcv(
            trading_symbol=trading_symbol,
            timeframe=timeframe_input,
            limit=limit_input
        )
    except Exception as e_fetch: # Catch any exception during fetch process
        logger.error(f"Error fetching market data for {trading_symbol} from {exchange_input}: {e_fetch}", exc_info=True)
        print(f"\nFailed to fetch market data. Error: {e_fetch}")

    if df_ohlcv is None or df_ohlcv.empty:
        print(f"\nCould not fetch OHLCV data for {trading_symbol} from {exchange_input}. Cannot perform analysis.")
    else:
        print(f"\nFetched {len(df_ohlcv)} candles for {trading_symbol}.")
        analyzer = TechnicalAnalyzer(df_ohlcv) # Initialize with fetched data
        
        # Generate indicators
        df_indicators = analyzer.generate_all_indicators() # This returns a copy

        if df_indicators is not None and not df_indicators.empty:
            print("\n--- DataFrame with Extended Technical Indicators (Tail) ---")
            pd.set_option("display.max_rows", 10)
            pd.set_option("display.max_columns", None) # Show all columns
            pd.set_option("display.width", 1000) 
            print(df_indicators.tail())
            
            print(f"\nTotal columns in indicators DataFrame: {len(df_indicators.columns)}")
            
            # --- Get LLM Overall Analysis (or fallback narrative) ---
            print("\n--- LLM Overall Technical Analysis ---")
            # The analyzer instance already has the indicators, so get_llm_overall_analysis will use them
            llm_analysis_result = analyzer.get_llm_overall_analysis(symbol=trading_symbol, timeframe=timeframe_input)
            
            if llm_analysis_result.get("error"):
                print(f"LLM Analysis Error: {llm_analysis_result['error']}")
                print("\n--- Fallback Indicator Narrative ---")
                narrative_fallback = llm_analysis_result.get("indicator_narrative_fallback", "Fallback narrative not available.")
                print(narrative_fallback)
            else:
                print(f"LLM Summary: {llm_analysis_result.get('summary_text')}")
                print(f"Overall Sentiment Category: {llm_analysis_result.get('overall_sentiment_category')}")
                print(f"Numeric Sentiment: {llm_analysis_result.get('numeric_sentiment')}")
        else:
            print("\nTechnical analysis (indicator generation) resulted in an empty or invalid DataFrame.")
            logger.warning(f"Indicator generation for {trading_symbol} returned None or empty DataFrame after processing {len(df_ohlcv)} candles.")

if __name__ == "__main__":
    # Ensure logger is configured if it wasn't by the utility import
    # This check ensures that if the custom get_logger fails, basic logging is still set up.
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
         # Re-initialize with basicConfig if no handlers or no stream handler exists
         # This primarily covers the case where the 'utils.logger' import failed entirely
         logging.basicConfig(
             level=logging.INFO,
             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
             handlers=[logging.StreamHandler()] # Ensure console output
         )
         logger.info("Fallback basic logger re-configured in __main__ of technical_analysis (if no stream handler was present).")

    try:
        asyncio.run(main_cli())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.")
    except Exception as e_main: # Catch any other unexpected errors in the main CLI execution
        logger.critical(f"Critical CLI Error in technical_analysis.py: {e_main}", exc_info=True)
        print(f"\nAn unexpected error occurred in the CLI: {e_main}")