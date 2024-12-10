#!/usr/bin/env python3

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

def get_assets_data(start_date='2009-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Downloading Bitcoin data...")
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
    print("Downloading NVIDIA data...")
    nvda_data = yf.download('NVDA', start=start_date, end=end_date)
    return btc_data, nvda_data

def calculate_metrics(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    
    metrics = {
        'mean_return': data['Daily Return'].mean(),
        'volatility': data['Daily Return'].std(),
        'sharpe_ratio': (data['Daily Return'].mean() / data['Daily Return'].std()) * np.sqrt(252)
    }
    return metrics

def plot_asset_prices(btc_data, nvda_data):
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data['Adj Close'], label='Bitcoin Price')
    plt.plot(nvda_data['Adj Close'], label='NVIDIA Price')
    plt.title('Bitcoin and NVIDIA Historical Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    
    # Guardar el gráfico
    plt.savefig('price_comparison.png')
    print("\nGráfico guardado como 'price_comparison.png'")
    
    # Mostrar el gráfico
    plt.show()

def display_metrics_comparison(btc_metrics, nvda_metrics, btc_data, nvda_data):
    btc_current_price = float(btc_data['Adj Close'].iloc[-1])
    nvda_current_price = float(nvda_data['Adj Close'].iloc[-1])
    
    metrics_df = pd.DataFrame({
        'Bitcoin': [
            f"${btc_current_price:,.2f}",
            f"{btc_metrics['mean_return']:.5f}",
            f"{btc_metrics['volatility']:.5f}",
            f"{btc_metrics['sharpe_ratio']:.2f}"
        ],
        'NVIDIA': [
            f"${nvda_current_price:,.2f}",
            f"{nvda_metrics['mean_return']:.5f}",
            f"{nvda_metrics['volatility']:.5f}",
            f"{nvda_metrics['sharpe_ratio']:.2f}"
        ]
    }, index=['Current Price', 'Mean Daily Return', 'Volatility (Std Dev)', 'Sharpe Ratio'])
    
    print("\nComparative Metrics:")
    print(metrics_df)
    
    return metrics_df

def main():
    try:
        # Get data and calculate metrics
        btc_data, nvda_data = get_assets_data()
        
        print("\nCalculating metrics...")
        btc_metrics = calculate_metrics(btc_data)
        nvda_metrics = calculate_metrics(nvda_data)

        # Plot the price comparison
        plot_asset_prices(btc_data, nvda_data)

        # Display metrics and get the DataFrame
        metrics_df = display_metrics_comparison(btc_metrics, nvda_metrics, btc_data, nvda_data)

        # Export to CSV
        csv_filename = 'comparative_metrics.csv'
        metrics_df.to_csv(csv_filename)
        print(f"\nMétricas guardadas en '{csv_filename}'")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())