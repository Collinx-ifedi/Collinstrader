# technical_analysis.py
import sys
import os
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1' # For compatibility if needed
import pandas as pd
import pandas_ta as ta
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
        # This is important as pandas_ta needs numeric inputs.
        initial_rows = len(df_processed)
        df_processed.dropna(subset=required_cols, inplace=True)
        if len(df_processed) < initial_rows:
            logger.info(f"Dropped {initial_rows - len(df_processed)} rows with NaN in essential OHLCV columns.")

        self.df_with_indicators = df_processed # This DataFrame will hold OHLCV + indicators

        # Define the comprehensive indicator strategy
        self.strategy = ta.Strategy(
            name="ComprehensiveIndicatorSet",
            description="Extensive set of indicators from Trend, Momentum, Volatility, and Volume categories using Pandas TA",
            ta=[
                # Trend Indicators
                {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 12}, {"kind": "ema", "length": 26}, {"kind": "ema", "length": 50}, {"kind": "ema", "length": 200},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9}, # MACD, MACDh, MACDs
                {"kind": "ichimoku", "tenkan": 9, "kijun": 26, "senkou": 52, "include_chikou": True}, # 5 columns
                {"kind": "adx", "length": 14}, # ADX, DMP, DMN
                {"kind": "aroon", "length": 14}, # AROOND, AROONU, AROONOSC
                {"kind": "psar"}, # PSARl, PSARs, PSARaf, PSARr
                {"kind": "stc", "tclength":23, "fastlength":10, "slowlength":50, "append": True}, # Schaff Trend Cycle
                {"kind": "vhf", "length": 28, "append": True}, # Vertical Horizontal Filter

                # Momentum Indicators
                {"kind": "rsi", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3}, # STOCHk, STOCHd
                {"kind": "stochrsi", "length": 14, "rsi_length": 14, "k": 3, "d": 3, "append": True}, # STOCHRSIk, STOCHRSId
                {"kind": "cci", "length": 20},
                {"kind": "roc", "length": 12},
                {"kind": "mom", "length": 10}, {"kind": "mom", "length": 20, "prefix": "MOM20", "append": True}, # Added longer Momentum
                {"kind": "ppo"}, # PPO, PPOh, PPOs
                {"kind": "slope", "length": 14}, {"kind": "slope", "length": 30, "prefix": "SLOPE30", "append": True},
                {"kind": "willr", "length": 14, "append": True}, # Williams %R

                # Volatility Indicators
                {"kind": "bbands", "length": 20, "std": 2}, # BBL, BBM, BBU, BBB, BBP
                {"kind": "atr", "length": 14},
                {"kind": "donchian", "lower_length": 20, "upper_length": 20}, # DCL, DCM, DCU
                {"kind": "kc", "length": 20, "scalar": 2, "mamode": "ema", "append": True}, # Keltner Channels (EMA based)
                {"kind": "chop", "length": 14, "append": True}, # Choppiness Index
                {"kind": "squeeze", "lazybear": False, "detailed": False, "append":True}, # TTM Squeeze (basic)
                
                # Volume Indicators
                {"kind": "obv"},
                {"kind": "cmf", "length": 20},
                {"kind": "mfi", "length": 14},
                {"kind": "efi", "length": 13, "append": True}, # Elder's Force Index
                {"kind": "vwap", "anchor": "D", "append": True}, # Daily VWAP
                {"kind": "vp", "width": 10, "append":True} # Volume Profile (basic, volume bars) - might add many columns
            ]
        )
        logger.info(f"TechnicalAnalyzer initialized. Strategy includes {len(self.strategy.ta)} indicator configurations.")


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
        if self.df_with_indicators.empty or not all(col in self.df_with_indicators.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.error("OHLCV DataFrame is empty or missing required columns for indicator generation.")
            return None
        
        # Check if sufficient data points for longest period indicators (e.g., SMA200)
        min_required_length = 200 # Based on SMA200 or similar long-period indicators
        if len(self.df_with_indicators) < min_required_length:
             logger.warning(f"Insufficient data ({len(self.df_with_indicators)} rows) for some long-period indicators (e.g., SMA200). Results may be sparse.")
        
        try:
            # Calculate indicators using the defined strategy. This appends to self.df_with_indicators.
            self.df_with_indicators.ta.strategy(self.strategy, timed=False, verbose=False)
            
            # Add Golden/Death Crosses
            self._add_golden_death_crosses() # Modifies self.df_with_indicators
            
            num_generated_cols = len(self.df_with_indicators.columns) - len(self.df_ohlcv_original.columns)
            logger.info(f"Successfully generated TA indicators. Added {num_generated_cols} columns.")
            return self.df_with_indicators.copy() # Return a copy of the full DataFrame
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}", exc_info=True)
            # Return the DataFrame with any indicators that were successfully calculated before the error
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
        sqz_on = latest.get('SQZ_ON') # Default column name from pandas_ta for squeeze on/off
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