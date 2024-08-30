import subprocess
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer



# Function to install necessary libraries
def install_libraries():
    subprocess.check_call(['pip', 'install', 'yfinance'])
    subprocess.check_call(['pip', 'install', 'pandas'])
    subprocess.check_call(['pip', 'install', 'numpy'])
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    subprocess.check_call(['pip', 'install', 'matplotlib'])


# Function to retrieve historical stock data from Yahoo Finance
def get_stock_data(ticker, start_date=None, end_date=None):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock


# Function to engineer features for stock prediction
def engineer_features(df):
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Calculate exponential moving averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # Calculate MACD
    df['MACD'] = df['EMA12'] - df['EMA26']

    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['SMA20'] + 2 * df['STD20']
    df['LowerBand'] = df['SMA20'] - 2 * df['STD20']

    # Calculate Price Rate of Change (ROC)
    n = 5
    df['ROC'] = df['Close'].pct_change(periods=n) * 100

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = np.where(df['Close'].diff() > 0, df['Volume'],
                         np.where(df['Close'].diff() < 0, -df['Volume'], 0)).cumsum()

    # Fill missing values
    df.fillna(method='bfill', inplace=True)

    return df


# Function to prepare data for model training
def prepare_data(df, target='Close'):
    df = engineer_features(df)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA10', 'MA20', 'EMA12', 'EMA26',
                'MACD', 'RSI', 'UpperBand', 'LowerBand', 'ROC', 'OBV']
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Function to train a Random Forest Regressor model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Function to make predictions using the trained model
def make_prediction(model, X):
    return model.predict(X)


# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")



# Function to get fundamental data
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        'P/E Ratio': info.get('forwardEps', np.nan) / info.get('forwardEps', np.nan),
        'Market Cap': info.get('marketCap', np.nan),
        'PE Ratio': info.get('trailingPE', np.nan),
        'Dividend Yield': info.get('dividendYield', np.nan),
        'Earnings Per Share': info.get('earningsPerShare', np.nan),
        'Total Revenue': info.get('totalRevenue', np.nan),
        'Gross Profit': info.get('grossProfit', np.nan),
        'Operating Margin': info.get('operatingMargins', np.nan),
        'Profit Margin': info.get('profitMargins', np.nan)
    }
    return fundamentals


# Main function to run the script
if __name__ == "__main__":
    li=["SHREECEM.BO", "RELIANCE.BO", "TCS.BO", "TIDEWATER.BO", "JUBLFOOD.BO"]
    install_libraries()

    #ticker = input("Enter the ticker symbol of the stock (e.g., AAPL): ").strip().upper()
    for ticker in li:
    # Retrieve maximum available historical data
        stock_data = get_stock_data(ticker)

    # Prepare data for model training
        X_train, X_test, y_train, y_test = prepare_data(stock_data)

    # Train the model
        model = train_model(X_train, y_train)

    # Make predictions for a future date (set as the last date of available data)
        future_date = datetime(2024, 8, 5)  # Replace with your desired future date

    # Get stock data up to the hardcoded future date
        future_data = get_stock_data(ticker, end_date=future_date)

        future_data = engineer_features(future_data)
        X_future = future_data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume',
                                     'MA5', 'MA10', 'MA20', 'EMA12', 'EMA26',
                                     'MACD', 'RSI', 'UpperBand', 'LowerBand', 'ROC', 'OBV']].values.reshape(1, -1)

    # Make prediction
        predicted_price = make_prediction(model, X_future)

        print(f"\nPredicted closing price for {ticker} on {future_date.date()}: ${predicted_price[0]:.2f}")

    # Evaluate model performance
        evaluate_model(model, X_test, y_test)

    # Print fundamental analysis
        fundamentals = get_fundamental_data(ticker)
        print("\nFundamental Analysis:")
        for key, value in fundamentals.items():
            print(f"{key}: {value}")

