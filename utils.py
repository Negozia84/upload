import json
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def save_optimized_params(asset: str, params: Dict) -> None:
    """Guarda parámetros optimizados en un archivo JSON."""
    try:
        filename = os.path.join(os.path.dirname(__file__), f"{asset.replace('/', '_')}_params.json")
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"{asset}: Parámetros guardados en '{filename}'")
    except Exception as e:
        logger.error(f"{asset}: Error al guardar parámetros: {e}")

def load_optimized_params(asset: str) -> Dict:
    """Carga parámetros optimizados desde un archivo JSON."""
    try:
        filename = os.path.join(os.path.dirname(__file__), f"{asset.replace('/', '_')}_params.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                params = json.load(f)
            logger.info(f"{asset}: Parámetros cargados desde '{filename}'")
            return params
        logger.debug(f"{asset}: No se encontró archivo de parámetros en '{filename}'")
        return {}
    except Exception as e:
        logger.error(f"{asset}: Error al cargar parámetros: {e}")
        return {}