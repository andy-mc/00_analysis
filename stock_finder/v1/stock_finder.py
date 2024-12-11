#!/usr/bin/env python3

# 1. Importar bibliotecas
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Diccionario con los nombres largos de las empresas
asset_names = {
    'BTC-USD': 'Bitcoin',
    'META': 'Meta Platforms, Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'TSLA': 'Tesla, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'ETH-USD': 'Ethereum',
    'GOOG': 'Alphabet Inc. (Google)',
    'AAPL': 'Apple Inc.',
    'MA': 'Mastercard Incorporated',
    'V': 'Visa Inc.',
    'MSFT': 'Microsoft Corporation',
    'XOM': 'Exxon Mobil Corporation',
    'UNH': 'UnitedHealth Group Incorporated',
    'JNJ': 'Johnson & Johnson'
}

# Métricas objetivo de Bitcoin y NVIDIA
target_metrics = {
    "Bitcoin": {
        "mean_return": 0.00275,
        "volatility": 0.02782,
        "sharpe_ratio": 1.57,
        "max_drawdown": -0.2618,
        "sortino_ratio": 2.62,
        "cagr": 0.8103
    },
    "NVIDIA": {
        "mean_return": 0.00491,
        "volatility": 0.03292,
        "sharpe_ratio": 2.37,
        "max_drawdown": -0.2705,
        "sortino_ratio": 3.78,
        "cagr": 1.9908
    }
}

# Calcular métricas de rendimiento
def calculate_metrics(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    mean_return = data['Daily Return'].mean()
    volatility = data['Daily Return'].std()
    sharpe_ratio = (mean_return / volatility) * np.sqrt(252)
    max_drawdown = (data['Cumulative Return'] / data['Cumulative Return'].cummax() - 1).min()
    sortino_ratio = calculate_sortino_ratio(data)
    cagr = calculate_cagr(data)
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'cagr': cagr
    }

def calculate_sortino_ratio(data):
    downside_returns = data['Daily Return'][data['Daily Return'] < 0]
    if downside_returns.empty:
        return np.nan  # Si no hay retornos negativos
    downside_std = downside_returns.std()
    sortino_ratio = (data['Daily Return'].mean() / downside_std) * np.sqrt(252)
    return sortino_ratio

def calculate_cagr(data):
    total_return = data['Cumulative Return'].iloc[-1]
    n_years = len(data) / 252  # Asumiendo 252 días hábiles al año
    cagr = total_return ** (1 / n_years) - 1
    return cagr

# Evaluar similitud de métricas
def similarity_score(metrics, target):
    score = 0
    for key in metrics:
        diff = abs(metrics[key] - target[key])
        normalized_diff = diff / max(abs(metrics[key]), abs(target[key]) + 1e-9)
        score += normalized_diff
    return score

# Main Script
def main():
    # Calcular fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    results = []

    print("Downloading data for assets...")
    for asset in asset_names:
        print(f"Processing {asset}...")
        try:
            data = yf.download(asset, start=start_date_str, end=end_date_str)
            if data.empty:
                print(f"No data for {asset}. Skipping...")
                continue

            # Calcular métricas
            metrics = calculate_metrics(data)
            current_price = data['Adj Close'].iloc[-1]  # Precio actual

            # Verificar que current_price es un escalar
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[-1]  # Seleccionar el último valor

            btc_similarity = similarity_score(metrics, target_metrics["Bitcoin"])
            nvda_similarity = similarity_score(metrics, target_metrics["NVIDIA"])
            
            # Agregar resultados
            results.append({
                'Asset': asset,
                'Company Name': asset_names[asset],
                'Current Price': float(current_price),
                'BTC Similarity': btc_similarity,
                'NVDA Similarity': nvda_similarity,
                **metrics
            })
        except Exception as e:
            print(f"Error processing {asset}: {e}")

    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results)

    # Verificar columnas antes de ordenar
    for column in ['Current Price', 'BTC Similarity', 'NVDA Similarity']:
        if results_df[column].dtype != 'float64' and results_df[column].dtype != 'int64':
            results_df[column] = pd.to_numeric(results_df[column], errors='coerce')

    # Ordenar por menor precio actual y luego por similitud
    results_df.sort_values(by=['Current Price', 'BTC Similarity', 'NVDA Similarity'], ascending=[True, True, True], inplace=True)

    # Exportar resultados
    results_df.to_csv('similar_assets_with_details.csv', index=False)
    print("\nResults saved to 'similar_assets_with_details.csv'")
    print(results_df.head(10))  # Mostrar los 10 activos más similares

if __name__ == "__main__":
    main()