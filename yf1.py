import yfinance as yf
import pandas as pd

# Define the ticker symbol
ticker_symbol = 'AAPL'

# Get data on this ticker
ticker_data = yf.Ticker(ticker_symbol)

# Get historical market data
hist = ticker_data.history(period="1y")  # You can change the period to "5d", "1mo", "1y", "5y", etc.

# Remove timezone information from all datetime columns
hist.index = hist.index.tz_convert(None)  # Ensure the index is timezone-unaware
hist.reset_index(inplace=True)  # Reset the index to convert the DatetimeIndex to a column
if 'Date' in hist.columns:
    hist['Date'] = hist['Date'].dt.tz_localize(None)  # Remove timezone from the 'Date' column if it exists

# Define the file name
file_name = f'{ticker_symbol}_historical_data.xlsx'

# Save the historical data to an Excel file
hist.to_excel(file_name, index=False)  # Save without the index to avoid any timezone issues

print(f"Historical data saved to {file_name}")
