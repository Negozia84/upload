import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def fetch_historical_data(exchange, symbol: str, timeframe: str, days: int = 90, retries: int = 3) -> Optional[pd.DataFrame]:
    """Descarga datos históricos del exchange con reintentos."""
    for attempt in range(retries):
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)
            api_symbol = symbol.split(':')[0]
            ohlcv = []
            limit = 1000
            
            while since < end_time:
                data = exchange.fetch_ohlcv(api_symbol, timeframe, since=since, limit=limit)
                if not data:
                    logger.warning(f"{symbol}: No hay más datos disponibles")
                    break
                ohlcv.extend(data)
                since = data[-1][0] + 1
                time.sleep(0.1)
            
            if not ohlcv:
                logger.warning(f"{symbol}: No se descargaron datos históricos")
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.drop_duplicates(subset=['timestamp'], inplace=True)
            logger.info(f"{symbol}: Descargadas {len(df)} velas desde {df['timestamp'].iloc[0]} hasta {df['timestamp'].iloc[-1]}")
            return df
        except Exception as e:
            logger.warning(f"{symbol}: Error en descarga histórica (intento {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                logger.error(f"{symbol}: Fallo tras {retries} intentos")
                return None

def fetch_real_time_data(exchange, symbol: str, timeframe: str, limit: int = 500, retries: int = 3) -> Optional[pd.DataFrame]:
    """Descarga datos en tiempo real con reintentos."""
    for attempt in range(retries):
        try:
            api_symbol = symbol.split(':')[0]
            ohlcv = exchange.fetch_ohlcv(api_symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.info(f"{symbol}: Descargadas {len(df)} velas en tiempo real")
            return df
        except Exception as e:
            logger.warning(f"{symbol}: Error en descarga en tiempo real (intento {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                logger.error(f"{symbol}: Fallo tras {retries} intentos")
                return None