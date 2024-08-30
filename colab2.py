import subprocess
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Install necessary libraries if not already installed
def install_libraries():
    subprocess.check_call(['pip', 'install', 'yfinance'])
    subprocess.check_call(['pip', 'install', 'pandas'])
    subprocess.check_call(['pip', 'install', 'numpy'])
    subprocess.check_call(['pip', 'install', 'scikit-learn'])

# Function to retrieve historical stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

# Function to engineer features for stock prediction
def engineer_features(df):
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()  # 5-day moving average
    df['MA10'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average

    # Calculate exponential moving averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA

    # Calculate MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA12'] - df['EMA26']

    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
    df['STD20'] = df['Close'].rolling(window=20).std()  # 20-day Standard Deviation
    df['UpperBand'] = df['SMA20'] + 2 * df['STD20']
    df['LowerBand'] = df['SMA20'] - 2 * df['STD20']

    # Calculate Price Rate of Change (ROC)
    n = 5  # Using a 5-day ROC
    df['ROC'] = df['Close'].pct_change(periods=n) * 100

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = np.where(df['Close'].diff() > 0, df['Volume'],
                         np.where(df['Close'].diff() < 0, -df['Volume'], 0)).cumsum()

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df

# Function to prepare data for model training
def prepare_data(df, target='Close'):
    # Select features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA10', 'MA20', 'EMA12', 'EMA26',
                'MACD', 'RSI', 'UpperBand', 'LowerBand', 'ROC', 'OBV']
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
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

if __name__ == "__main__":
    # Install necessary libraries if not already installed
    install_libraries()

    # Ask user for input (ticker symbol)
    ticker = input("Enter the ticker symbol of the stock (e.g., AAPL): ").strip().upper()

    # Set start and end dates for historical data
    start_date = '2019-01-01'
    end_date = '2024-06-06'

    # Get historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Engineer features
    stock_data = engineer_features(stock_data)

    # Prepare data for model training
    X_train, X_test, y_train, y_test = prepare_data(stock_data)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions for the hardcoded future date (22nd July 2024)
    last_data_point = stock_data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume',
                                           'MA5', 'MA10', 'MA20', 'EMA12', 'EMA26',
                                           'MACD', 'RSI', 'UpperBand', 'LowerBand', 'ROC', 'OBV']].values.reshape(1, -1)

    # Make prediction
    predicted_price = make_prediction(model, last_data_point)

    print(f"\nPredicted closing price for {ticker} on July 22, 2024: ${predicted_price[0]:.2f}")

    # Evaluate model performance
    evaluate_model(model, X_test, y_test)
