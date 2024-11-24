import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

symbols = ['GOOGL', 'AAPL', 'AMZN']
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

data = yf.download(symbols, start=start_date_str, end=end_date_str)
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
fig.suptitle('Stock Prices - Last Month (Open, Close, High, Low)', fontsize=16)

for i, symbol in enumerate(symbols):
    axs[i].plot(data.index, data['Open'][symbol], label='Open', color='green')
    axs[i].plot(data.index, data['Close'][symbol], label='Close', color='red')
    axs[i].plot(data.index, data['High'][symbol], label='High', color='blue')
    axs[i].plot(data.index, data['Low'][symbol], label='Low', color='orange')
    
    axs[i].set_title(f'{symbol} Stock Price')
    axs[i].set_ylabel('Price (USD)')
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Date')
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('stock_prices_ohlc_last_month.png')
plt.show()