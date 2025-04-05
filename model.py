import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def prepare_features_and_labels(df: pd.DataFrame, label_window: int = 10) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Prepara características y etiquetas para el modelo."""
    features_list = ['sma', 'rsi', 'atr', 'adx', 'ema_short', 'ema_long', 'macd', 'macd_signal', 'volatility', 'volume_sma']
    try:
        features = df[features_list].copy()
        if len(df) - label_window <= 0:
            logger.warning("Datos insuficientes para preparar etiquetas")
            return None, None
        
        labels = (df['close'].shift(-label_window) > df['close'] * 1.03).astype(int)
        df_combined = pd.concat([features, labels.rename('label')], axis=1).dropna()
        
        if len(df_combined) < 10:
            logger.warning("Datos insuficientes tras limpieza")
            return None, None
        
        pos, neg = df_combined['label'].eq(1), df_combined['label'].eq(0)
        if pos.sum() == 0 or neg.sum() == 0:
            logger.warning("Clases desbalanceadas, usando datos sin ajustar")
            return df_combined[features_list], df_combined['label']
        
        minority, majority = (pos, neg) if pos.sum() < neg.sum() else (neg, pos)
        minority_upsampled = df_combined[minority].sample(n=majority.sum(), replace=True, random_state=42)
        df_balanced = pd.concat([minority_upsampled, df_combined[majority]])
        logger.info(f"Features y labels preparados: {len(df_balanced)} filas")
        return df_balanced[features_list], df_balanced['label']
    except Exception as e:
        logger.error(f"Error al preparar features y labels: {e}")
        return None, None

def train_model(features: pd.DataFrame, labels: pd.Series, filename: str) -> Optional[LGBMClassifier]:
    """Entrena y guarda un modelo LGBM."""
    try:
        if len(features) < 10 or len(labels) < 10:
            logger.warning(f"Datos insuficientes para entrenar modelo: {len(features)} filas")
            return None
        model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model.fit(features, labels)
        joblib.dump(model, filename)
        logger.info(f"Modelo guardado en '{filename}'")
        return model
    except Exception as e:
        logger.error(f"Error al entrenar modelo: {e}")
        return None

def generate_signals(df: pd.DataFrame, model: LGBMClassifier) -> pd.DataFrame:
    """Genera señales de trading basadas en predicciones del modelo."""
    features_list = ['sma', 'rsi', 'atr', 'adx', 'ema_short', 'ema_long', 'macd', 'macd_signal', 'volatility', 'volume_sma']
    try:
        features = df[features_list].copy()
        probs = model.predict_proba(features)[:, 1]
        df['signal'] = np.select([probs > 0.6, probs < 0.4], [1, -1], default=0)
        logger.info(f"Señales generadas para {len(df)} filas")
        return df
    except Exception as e:
        logger.error(f"Error al generar señales: {e}")
        return df  # Devuelve el DataFrame sin cambios si falla