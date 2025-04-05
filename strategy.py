import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import optuna
from config import CONFIG  # Importamos CONFIG para usar valores globales
from indicators import calculate_indicators
from model import prepare_features_and_labels, train_model, generate_signals

logger = logging.getLogger(__name__)

# ... (importaciones previas en strategy.py)

def backtest_strategy(df: pd.DataFrame, config: Dict, initial_capital: float, execute_trades: bool = False, exchange=None) -> Tuple[float, List[Dict]]:
    if execute_trades and exchange is None:
        logger.error("Se requiere un objeto exchange para trading en vivo")
        return initial_capital, []
    
    capital, position, entry_price, stop_loss, trailing_stop = initial_capital, 0, 0.0, 0.0, 0.0
    min_capital, trades = initial_capital, []
    symbol = config.get('symbol', 'Unknown')
    api_symbol = symbol.split(':')[0]
    
    for i, row in df.iterrows():
        if capital < initial_capital * (1 - CONFIG['trading'].get('max_drawdown', 0.5)):
            logger.warning(f"{symbol}: Máximo drawdown alcanzado")
            break
        if capital < initial_capital * (1 - 0.2):
            logger.warning(f"{symbol}: Drawdown intermedio (20%) alcanzado. Pausando.")
            break
        
        close, signal, atr, rsi = row['close'], row['signal'], row['atr'], row['rsi']
        position_size = (initial_capital * config['risk_per_trade'] * CONFIG['trading'].get('leverage', 2)) / close
        position_size = min(position_size, capital / close)  # Limitar al capital disponible
        
        if position == 0 and capital > 0:
            if signal == 1 and config['rsi_threshold'] < rsi < config['rsi_overbought']:
                position, entry_price = 1, close
                stop_loss = close * (1 - config.get('sl_atr_multiplier', 2) * atr / close)
                if execute_trades:
                    try:
                        order = exchange.create_market_buy_order(api_symbol, position_size)
                        entry_price = float(order['price']) if order['price'] else close
                        capital -= position_size * entry_price * CONFIG['trading'].get('fee_rate', 0.0006)
                        logger.info(f"{symbol}: Orden real - Compra long a {entry_price}, Tamaño: {position_size}")
                    except Exception as e:
                        logger.error(f"{symbol}: Error al abrir long: {e}")
                        continue
                else:
                    capital -= position_size * close * CONFIG['trading'].get('fee_rate', 0.0006)
                trades.append({'entry_time': row['timestamp'], 'entry_price': entry_price, 'type': 'BUY_LONG'})
                logger.info(f"{symbol}: Compra long a {entry_price}")
            
            elif signal == -1 and rsi > config['rsi_threshold']:
                position, entry_price = -1, close
                stop_loss = close * (1 + config.get('sl_atr_multiplier', 2) * atr / close)
                if execute_trades:
                    try:
                        order = exchange.create_market_sell_order(api_symbol, position_size)
                        entry_price = float(order['price']) if order['price'] else close
                        capital -= position_size * entry_price * CONFIG['trading'].get('fee_rate', 0.0006)
                        logger.info(f"{symbol}: Orden real - Venta short a {entry_price}, Tamaño: {position_size}")
                    except Exception as e:
                        logger.error(f"{symbol}: Error al abrir short: {e}")
                        continue
                else:
                    capital -= position_size * close * CONFIG['trading'].get('fee_rate', 0.0006)
                trades.append({'entry_time': row['timestamp'], 'entry_price': entry_price, 'type': 'SELL_SHORT'})
                logger.info(f"{symbol}: Venta short a {entry_price}")

        elif position != 0:
            profit_pct = (close - entry_price) / entry_price if position == 1 else (entry_price - close) / entry_price
            exit_condition = signal == 0 or (position == 1 and row['low'] <= stop_loss) or (position == -1 and row['high'] >= stop_loss)
            
            breakeven_threshold = config.get('breakeven_threshold', 0.015)
            if position == 1 and profit_pct >= breakeven_threshold:
                stop_loss = max(stop_loss, entry_price * (1 + CONFIG['trading'].get('fee_rate', 0.0006) * 2))
            elif position == -1 and profit_pct >= breakeven_threshold:
                stop_loss = min(stop_loss, entry_price * (1 - CONFIG['trading'].get('fee_rate', 0.0006) * 2))
            
            if position == 1:
                trailing_stop = max(trailing_stop, close * (1 - config['trailing_distance'])) if trailing_stop != 0 else entry_price * (1 + config['target_gain'])
            elif position == -1:
                trailing_stop = min(trailing_stop, close * (1 + config['trailing_distance'])) if trailing_stop != 0 else entry_price * (1 - config['target_gain'])
            
            if position == 1 and row['low'] <= trailing_stop or position == -1 and row['high'] >= trailing_stop:
                exit_condition = True
            
            if exit_condition:
                exit_price = close if signal == 0 else (stop_loss if position == 1 and row['low'] <= stop_loss else stop_loss if position == -1 and row['high'] >= stop_loss else trailing_stop)
                if execute_trades:
                    try:
                        order = exchange.create_market_sell_order(api_symbol, position_size) if position == 1 else exchange.create_market_buy_order(api_symbol, position_size)
                        exit_price = float(order['price']) if order['price'] else exit_price
                        profit = position_size * (exit_price - entry_price) * position - 2 * position_size * exit_price * CONFIG['trading'].get('fee_rate', 0.0006)
                        profit = min(profit, capital)  # Limitar profit al capital disponible
                        capital += profit
                        logger.info(f"{symbol}: Orden real - Salida a {exit_price}, Profit: {profit:.2f}")
                    except Exception as e:
                        logger.error(f"{symbol}: Error al cerrar posición: {e}")
                        continue
                else:
                    profit = position_size * (exit_price - entry_price) * position - 2 * position_size * exit_price * CONFIG['trading'].get('fee_rate', 0.0006)
                    profit = min(profit, capital)  # Limitar profit al capital disponible
                    capital += profit
                trades[-1].update({'exit_time': row['timestamp'], 'exit_price': exit_price, 'profit': profit})
                logger.info(f"{symbol}: Salida a {exit_price}, Profit: {profit:.2f}")
                position = 0
                trailing_stop = 0.0
        
        min_capital = min(min_capital, capital)
    
    # ... (cierre forzado y métricas finales sin cambios)
    

# ... (optimize_parameters igual)
    
    # Cierre forzado de posición abierta
    if position != 0:
        close = df['close'].iloc[-1]
        exit_price = close
        profit = position_size * (exit_price - entry_price) * position - 2 * position_size * exit_price * CONFIG['trading'].get('fee_rate', 0.0006)
        capital += profit
        trades[-1].update({'exit_time': df['timestamp'].iloc[-1], 'exit_price': exit_price, 'profit': profit})
        logger.info(f"{symbol}: Cierre forzado de posición a {exit_price}, Profit: {profit:.2f}")
    
    # Métricas finales
    total_profit, total_trades = capital - initial_capital, len(trades)
    wins = sum(1 for t in trades if t.get('profit', 0) > 0)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    profit_factor = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0) / abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0)) if any(t.get('profit', 0) < 0 for t in trades) else float('inf')
    max_drawdown_pct = (initial_capital - min_capital) / initial_capital * 100 if initial_capital > 0 else 0
    
    logger.info(f"{symbol}: Capital Final: {capital:.2f}, Profit Total: {total_profit:.2f}, Trades: {total_trades}, Win Rate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}, Max Drawdown: {max_drawdown_pct:.2f}%")
    return capital, trades

def optimize_parameters(exchange, symbol: str, timeframe: str, df: pd.DataFrame, initial_capital: float, n_trials: int = 20) -> Tuple[Dict, float, List[Dict]]:
    """Optimiza parámetros de la estrategia usando Optuna."""
    def objective(trial):
        params = {
            'sma_period': trial.suggest_int('sma_period', 50, 300),
            'rsi_period': trial.suggest_int('rsi_period', 5, 20),
            'rsi_threshold': trial.suggest_float('rsi_threshold', 30, 60),
            'rsi_overbought': trial.suggest_float('rsi_overbought', 70, 90),
            'target_gain': trial.suggest_float('target_gain', 0.03, 0.10),
            'sl_atr_multiplier': trial.suggest_float('sl_atr_multiplier', 1.5, 3.0),
            'trailing_distance': trial.suggest_float('trailing_distance', 0.01, 0.03),
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            'breakeven_threshold': trial.suggest_float('breakeven_threshold', 0.01, 0.03),
            'symbol': symbol
        }
        df_opt = calculate_indicators(df.copy(), params['sma_period'], params['rsi_period'])
        features, labels = prepare_features_and_labels(df_opt)
        if features is None or labels is None:
            return -float('inf')
        model = train_model(features, labels, f"{symbol.replace('/', '_')}_model.pkl")
        if model is None:
            return -float('inf')
        df_opt = generate_signals(df_opt, model)
        final_capital, _ = backtest_strategy(df_opt, params, initial_capital)
        return final_capital - initial_capital
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['symbol'] = symbol
    logger.info(f"{symbol}: Mejores parámetros encontrados - Profit: {study.best_value:.2f}, Params: {best_params}")
    return best_params, study.best_value, []