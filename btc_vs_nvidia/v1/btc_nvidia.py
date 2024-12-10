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
    start_date = end_date - timedelta(days=365)  # 365 días = 1 año
    
    # Convertir fechas a string en formato YYYY-MM-DD
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\nObteniendo datos desde {start_date_str} hasta {end_date_str}")
    
    print("Downloading Bitcoin data...")
    btc_data = yf.download('BTC-USD', start=start_date_str, end=end_date_str)
    print("Downloading NVIDIA data...")
    nvda_data = yf.download('NVDA', start=start_date_str, end=end_date_str)
    return btc_data, nvda_data

def calculate_metrics(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    
    metrics = {
        'mean_return': data['Daily Return'].mean(),
        'volatility': data['Daily Return'].std(),
        'sharpe_ratio': (data['Daily Return'].mean() / data['Daily Return'].std()) * np.sqrt(252)
    }
    return metrics

def normalize_prices(data):
    # Normalizar los precios: dividir cada precio diario entre el precio del primer día
    data['Normalized Price'] = data['Adj Close'] / data['Adj Close'].iloc[0]
    return data

def plot_normalized_prices(btc_data, nvda_data):
    # Graficar los precios normalizados
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data.index, btc_data['Normalized Price'], label='Bitcoin (Normalized)')
    plt.plot(nvda_data.index, nvda_data['Normalized Price'], label='NVIDIA (Normalized)')
    plt.title('Normalized Prices: Bitcoin vs NVIDIA (Last Year)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(alpha=0.5)
    
    # Guardar la gráfica
    plt.savefig('normalized_price_comparison_last_year.png')
    print("\nGráfico de precios normalizados guardado como 'normalized_price_comparison_last_year.png'")
    
    # Mostrar la gráfica
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
        
        # Normalize prices
        btc_data = normalize_prices(btc_data)
        nvda_data = normalize_prices(nvda_data)

        # Plot normalized prices
        plot_normalized_prices(btc_data, nvda_data)

        # Display metrics and get the DataFrame
        metrics_df = display_metrics_comparison(btc_metrics, nvda_metrics, btc_data, nvda_data)

        # Export to CSV
        csv_filename = 'comparative_metrics_last_year.csv'
        metrics_df.to_csv(csv_filename)
        print(f"\nMétricas guardadas en '{csv_filename}'")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())