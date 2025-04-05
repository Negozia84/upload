import pandas as pd
import talib
import logging

logger = logging.getLogger(__name__)

def calculate_indicators(df: pd.DataFrame, sma_period: int, rsi_period: int) -> pd.DataFrame:
    """Calcula indicadores técnicos para el DataFrame."""
    try:
        df['sma'] = talib.SMA(df['close'], timeperiod=sma_period)
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['ema_short'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=50)
        macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'], df['macd_signal'] = macd, signal
        df['volatility'] = df['close'].rolling(window=14, min_periods=1).std()
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        logger.debug(f"Indicadores calculados para {len(df)} filas")
        return df
    except Exception as e:
        logger.error(f"Error al calcular indicadores: {e}")
        return df  # Devuelve el DataFrame sin cambios si falla