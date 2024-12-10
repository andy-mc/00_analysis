#!/usr/bin/env python3

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

def get_assets_data():
    # Calcular las fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Último año
    
    # Convertir fechas a string
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\nObteniendo datos desde {start_date_str} hasta {end_date_str}")
    
    # Descargar datos de Bitcoin y NVIDIA
    print("Downloading Bitcoin data...")
    btc_data = yf.download('BTC-USD', start=start_date_str, end=end_date_str)
    print("Downloading NVIDIA data...")
    nvda_data = yf.download('NVDA', start=start_date_str, end=end_date_str)
    
    return btc_data, nvda_data

def calculate_metrics(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    
    metrics = {
        'mean_return': data['Daily Return'].mean(),
        'volatility': data['Daily Return'].std(),
        'sharpe_ratio': (data['Daily Return'].mean() / data['Daily Return'].std()) * np.sqrt(252),
        'max_drawdown': (data['Cumulative Return'] / data['Cumulative Return'].cummax() - 1).min(),
        'sortino_ratio': calculate_sortino_ratio(data),
        'cagr': calculate_cagr(data),
    }
    return metrics

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

def normalize_prices(data):
    data['Normalized Price'] = data['Adj Close'] / data['Adj Close'].iloc[0]
    return data

def plot_metrics(data, title, y_label, column, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column], label=title)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(filename)
    plt.show()

def plot_normalized_prices(btc_data, nvda_data):
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data.index, btc_data['Normalized Price'], label='Bitcoin (Normalized)')
    plt.plot(nvda_data.index, nvda_data['Normalized Price'], label='NVIDIA (Normalized)')
    plt.title('Normalized Prices: Bitcoin vs NVIDIA (Last Year)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig('normalized_price_comparison_last_year.png')
    plt.show()

def display_metrics_comparison(btc_metrics, nvda_metrics, btc_data, nvda_data):
    btc_current_price = float(btc_data['Adj Close'].iloc[-1])
    nvda_current_price = float(nvda_data['Adj Close'].iloc[-1])
    
    metrics_df = pd.DataFrame({
        'Metric': [
            'Current Price', 'Mean Daily Return', 'Volatility (Std Dev)', 
            'Sharpe Ratio', 'Max Drawdown', 'Sortino Ratio', 'CAGR'
        ],
        'Bitcoin': [
            f"${btc_current_price:,.2f}",
            f"{btc_metrics['mean_return']:.5f}",
            f"{btc_metrics['volatility']:.5f}",
            f"{btc_metrics['sharpe_ratio']:.2f}",
            f"{btc_metrics['max_drawdown']:.2%}",
            f"{btc_metrics['sortino_ratio']:.2f}",
            f"{btc_metrics['cagr']:.2%}",
        ],
        'NVIDIA': [
            f"${nvda_current_price:,.2f}",
            f"{nvda_metrics['mean_return']:.5f}",
            f"{nvda_metrics['volatility']:.5f}",
            f"{nvda_metrics['sharpe_ratio']:.2f}",
            f"{nvda_metrics['max_drawdown']:.2%}",
            f"{nvda_metrics['sortino_ratio']:.2f}",
            f"{nvda_metrics['cagr']:.2%}",
        ],
    })
    
    print("\nComparative Metrics:")
    print(metrics_df)
    return metrics_df

def main():
    try:
        # Obtener datos y calcular métricas
        btc_data, nvda_data = get_assets_data()
        
        print("\nCalculating metrics...")
        btc_data = normalize_prices(btc_data)
        nvda_data = normalize_prices(nvda_data)
        btc_metrics = calculate_metrics(btc_data)
        nvda_metrics = calculate_metrics(nvda_data)

        # Graficar precios normalizados
        plot_normalized_prices(btc_data, nvda_data)

        # Graficar métricas individuales
        plot_metrics(btc_data, 'Cumulative Return - Bitcoin', 'Cumulative Return', 'Cumulative Return', 'btc_cumulative_return.png')
        plot_metrics(nvda_data, 'Cumulative Return - NVIDIA', 'Cumulative Return', 'Cumulative Return', 'nvda_cumulative_return.png')

        # Mostrar y exportar métricas
        metrics_df = display_metrics_comparison(btc_metrics, nvda_metrics, btc_data, nvda_data)
        metrics_df.to_csv('comparative_metrics_with_new_indicators.csv', index=False)
        print("\nMetrics saved to 'comparative_metrics_with_new_indicators.csv'")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())