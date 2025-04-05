import ccxt
import logging
import time
import pandas as pd
import numpy as np  
from config import CONFIG, VERSION
from data import fetch_historical_data, fetch_real_time_data
from indicators import calculate_indicators
from model import prepare_features_and_labels, train_model, generate_signals
from strategy import backtest_strategy, optimize_parameters
from utils import save_optimized_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('trading.log')]
)
logger = logging.getLogger(__name__)

exchange = getattr(ccxt, CONFIG['exchange']['name'])({
    'apiKey': CONFIG['exchange']['apiKey'],
    'secret': CONFIG['exchange']['secret'],
    'password': CONFIG['exchange']['password'],
    'enableRateLimit': True,
    'options': CONFIG['exchange']['options']
})

def run_trading(live: bool = False, assets: list = None):
    """Ejecuta el bot de trading en modo simulación o en vivo."""
    mode = "live" if live else "backtest"
    logger.info(f"Iniciando trading bot - Modo: {mode}, Versión: {VERSION}")
    exchange.load_markets()
    
    all_assets = CONFIG['assets'].keys()
    if assets:
        target_assets = [asset for asset in assets if asset in all_assets]
        if not target_assets:
            logger.warning(f"Los activos especificados ({assets}) no están en la configuración. Usando todos.")
            target_assets = all_assets
    else:
        target_assets = all_assets
    
    logger.info(f"Activos seleccionados: {list(target_assets)}")
    optimized_params = {asset: load_optimized_params(asset) for asset in target_assets}
    optimization_counter = {asset: 0 for asset in target_assets}
    
    # Fase inicial
    for asset in target_assets:
        symbol = asset.split('-')[0]
        timeframe = CONFIG['assets'][asset]['timeframe']
        df = fetch_historical_data(exchange, symbol, timeframe)
        if df is None or len(df) < 400:
            logger.error(f"{asset}: Datos insuficientes para inicialización")
            continue
        
        if not optimized_params[asset]:
            logger.info(f"{asset}: Realizando optimización inicial...")
            best_params, best_profit, _ = optimize_parameters(exchange, symbol, timeframe, df, CONFIG['trading']['initial_capital'], n_trials=10)
            if not best_params:
                logger.error(f"{asset}: Fallo en optimización inicial")
                continue
            optimized_params[asset] = best_params
            save_optimized_params(asset, best_params)
        
        df_opt = calculate_indicators(df.copy(), optimized_params[asset]['sma_period'], optimized_params[asset]['rsi_period'])
        features, labels = prepare_features_and_labels(df_opt)
        model = train_model(features, labels, f"{symbol.replace('/', '_')}_model.pkl")
        if model:
            df_opt = generate_signals(df_opt, model)
            final_capital, _ = backtest_strategy(df_opt, optimized_params[asset], CONFIG['trading']['initial_capital'], execute_trades=live, exchange=exchange if live else None)
            logger.info(f"{asset}: {mode.capitalize()} inicial - Capital Final: {final_capital:.2f}")
    
    # Bucle en tiempo real
    if live:
        while True:
            try:
                for asset in optimized_params.keys():
                    symbol = asset.split('-')[0]
                    timeframe = CONFIG['assets'][asset]['timeframe']
                    config_asset = optimized_params[asset]
                    optimization_counter[asset] += 1
                    
                    if optimization_counter[asset] % 288 == 0:
                        logger.info(f"{asset}: Reoptimizando parámetros...")
                        df = fetch_historical_data(exchange, symbol, timeframe)
                        if df is not None:
                            best_params, best_profit, _ = optimize_parameters(exchange, symbol, timeframe, df, CONFIG['trading']['initial_capital'], n_trials=5)
                            if best_params:
                                optimized_params[asset] = best_params
                                save_optimized_params(asset, best_params)
                    
                    historical_data = fetch_historical_data(exchange, symbol, timeframe)
                    new_data = fetch_real_time_data(exchange, symbol, timeframe)
                    if new_data is None or historical_data is None:
                        logger.warning(f"{asset}: Fallo en descarga de datos")
                        time.sleep(60)
                        continue
                    
                    df_combined = pd.concat([historical_data, new_data]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    df_combined = calculate_indicators(df_combined, config_asset['sma_period'], config_asset['rsi_period'])
                    features, labels = prepare_features_and_labels(df_combined)
                    if features is None or labels is None:
                        logger.warning(f"{asset}: Fallo en preparación de datos")
                        time.sleep(60)
                        continue
                    
                    model = train_model(features, labels, f"{symbol.replace('/', '_')}_model.pkl")
                    if model is None:
                        time.sleep(60)
                        continue
                    
                    df_combined = generate_signals(df_combined, model)
                    final_capital, trades = backtest_strategy(df_combined, config_asset, CONFIG['trading']['initial_capital'], execute_trades=live, exchange=exchange)
                    save_optimized_params(asset, config_asset)
                    time.sleep(CONFIG['assets'][asset]['candle_duration'])
            
            except KeyboardInterrupt:
                logger.info("Bot detenido por el usuario")
                break
            except Exception as e:
                logger.error(f"Error crítico: {e}")
                time.sleep(60)

if __name__ == "__main__":
    run_trading()