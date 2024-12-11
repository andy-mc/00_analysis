#!/usr/bin/env python3

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Función para obtener todos los componentes del S&P 500
def get_sp500_components():
    print("Fetching S&P 500 components...")
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    tickers = sp500_table['Symbol'].tolist()
    company_names = sp500_table['Security'].tolist()
    return dict(zip(tickers, company_names))

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
        return np.nan
    downside_std = downside_returns.std()
    sortino_ratio = (data['Daily Return'].mean() / downside_std) * np.sqrt(252)
    return sortino_ratio

def calculate_cagr(data):
    total_return = data['Cumulative Return'].iloc[-1]
    n_years = len(data) / 252
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
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Obtener componentes del S&P 500
    asset_names = get_sp500_components()

    if not asset_names:
        print("No assets found. Exiting...")
        return

    results = []

    print("Downloading data for assets...")
    for asset, long_name in asset_names.items():
        print(f"Processing {asset} ({long_name})...")
        try:
            data = yf.download(asset, start=start_date_str, end=end_date_str)
            if data.empty:
                print(f"No data for {asset}. Skipping...")
                continue

            metrics = calculate_metrics(data)
            current_price = data['Adj Close'].iloc[-1]
            btc_similarity = similarity_score(metrics, target_metrics["Bitcoin"])
            nvda_similarity = similarity_score(metrics, target_metrics["NVIDIA"])

            results.append({
                'Asset': asset,
                'Company Name': long_name,
                'Current Price': float(current_price),
                'BTC Similarity': btc_similarity,
                'NVDA Similarity': nvda_similarity,
                **metrics
            })
        except Exception as e:
            print(f"Error processing {asset}: {e}")

    results_df = pd.DataFrame(results)

    for column in ['Current Price', 'BTC Similarity', 'NVDA Similarity']:
        if results_df[column].dtype != 'float64' and results_df[column].dtype != 'int64':
            results_df[column] = pd.to_numeric(results_df[column], errors='coerce')

    results_df.sort_values(by=['Current Price', 'BTC Similarity', 'NVDA Similarity'], ascending=[True, True, True], inplace=True)

    results_df.to_csv('sp500_similar_assets.csv', index=False)
    print("\nResults saved to 'sp500_similar_assets.csv'")
    print(results_df.head(10))

if __name__ == "__main__":
    main()