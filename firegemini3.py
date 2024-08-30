import os

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, render_template, request
import google.generativeai as genai


from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


import logging
app = Flask(__name__)
CORS(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('D:\\Desktop\\Projects\\ML\\oai\\money-3f17d-firebase-adminsdk-acnwl-9b1bf13ea8.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://money-3f17d-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Firebase database reference
database_reference = db.reference('Users')
app.secret_key = os.urandom(24)
@app.route('/')
def home():
    # Serve the login page
    return render_template('index4.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        # Serve the signup page
        return render_template('index3.html')

    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')
        age = data.get('age')
        profession = data.get('profession')

        if not username or not password or not age:
            return jsonify({"error": "All fields are required"}), 400

        try:
            age = int(age)
            if age <= 0:
                raise ValueError()
        except ValueError:
            return jsonify({"error": "Please enter a valid age"}), 400

        user_ref = db.reference('Users')
        user_ref.push({
            'username': username,
            'password': password,
            'age': age,
            'profession': profession
        })

        return jsonify({"success": True, "message": "Sign-up successful!"}), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Simple validation
    if not username or not password:
        return jsonify({"message": "Both fields are required"}), 400

    # Query Firebase Realtime Database
    query_ref = database_reference.order_by_child('username').equal_to(username)
    users = query_ref.get()

    if users:
        for user_id, user_data in users.items():
            if user_data.get('password') == password:
                # Login successful
                session['user_details'] = {
                    'username': username,
                    'age': user_data.get('age', 'N/A'),
                    'profession': user_data.get('profession', 'N/A')
                }
                session['chat_history'] = []  # Initialize chat history

                return jsonify({"message": "Correct login", "user_id": user_id}), 200

    return jsonify({"message": "Invalid username or password"}), 401


@app.route('/profile/<user_id>', methods=['GET', 'POST'])
def profile(user_id):
    if request.method == 'GET':
        # Fetch user data
        user_data = database_reference.child(user_id).get()
        if user_data:
            return render_template('index5.html', user=user_data, user_id=user_id)
        return jsonify({"message": "User not found"}), 404

    if request.method == 'POST':
        try:
            data = request.json
            password = data.get('password')
            age = data.get('age')

            if not password or not age:
                return jsonify({"error": "All fields are required"}), 400

            age = int(age)
            if age <= 0:
                return jsonify({"error": "Please enter a valid age"}), 400

            updated_user = {
                'password': password,
                'age': age
            }

            # Check if username needs to be updated
            username = data.get('username')
            if username:
                updated_user['username'] = username

            existing_user = database_reference.child(user_id).get()
            if existing_user:
                database_reference.child(user_id).update(updated_user)
                return jsonify({"success": True, "message": "Profile updated successfully!"}), 200
            else:
                return jsonify({"error": "User not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

#gemini now
api_key = 'AIzaSyDOHtE9TTov4mmdeJ1HwcXkvWPpVkzuWxU'

genai.configure(api_key=api_key)

# Create the model
generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        #user_input = request.form['user_input']  # This line alone is sufficient
        user_input = request.form.get('user_input')
        chat_history = session.get('chat_history', [])

        # Include user details if available
        user_details = session.get('user_details', {})
        if user_details:
            chat_history.append({
                'role': 'user',
                'parts': [{'text': f"User details: {user_details}"}]
            })

        # Append user input to the chat history
        chat_history.append({
            'role': 'user',
            'parts': [{'text': user_input}]
        })

        # Start or continue the chat session
        chat_session = model.start_chat(history=chat_history)
        response = chat_session.send_message(user_input)

        # Append the assistant's response to the chat history
        chat_history.append({
            'role': 'model',
            'parts': [{'text': response.text}]
        })

        # Save the updated chat history to the session
        session['chat_history'] = chat_history
        session.modified = True  # Ensure the session is marked as modified

        return render_template('index.html', user_input=user_input, response=response.text)

    return render_template('index.html')




# stock code  -----------------------

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    info = stock.info
    return hist, info


# Fundamental Analysis Tools
def calculate_fundamentals(info):
    pe_ratio = info.get("trailingPE", np.nan)
    eps = info.get("trailingEps", np.nan)
    de_ratio = info.get("debtToEquity", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    dividend_yield = info.get("dividendYield", np.nan) * 100 if info.get("dividendYield", np.nan) else np.nan
    market_cap = info.get("marketCap", np.nan)
    beta = info.get("beta", np.nan)
    return pe_ratio, eps, de_ratio, roe, dividend_yield, market_cap, beta


# Technical Analysis Tools
def calculate_technicals(hist):
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    hist['BB_upper'], hist['BB_lower'] = compute_bollinger_bands(hist['Close'])
    hist['ATR'] = compute_atr(hist)
    rsi = compute_rsi(hist['Close'])
    macd, signal, macd_histogram = compute_macd(hist['Close'])
    stochastic_k, stochastic_d = compute_stochastic_oscillator(hist)
    momentum = hist['Close'].pct_change().fillna(0)
    return hist, hist['MA50'], hist['MA200'], rsi, macd, signal, macd_histogram, stochastic_k, stochastic_d, momentum


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd - signal
    return macd, signal, macd_histogram


def compute_stochastic_oscillator(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    return k, d


def compute_bollinger_bands(data, window=20, num_std=2):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, lower_band


def compute_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


# Weight assignment
def assign_weights(fundamentals, technicals):
    fundamental_weights = [0.2] * 7  # 7 weights for 7 fundamentals
    technical_weights = [0.1] * 8  # 8 weights for 8 technicals

    fundamental_score = np.nansum(np.array(fundamentals) * np.array(fundamental_weights))
    technical_score = np.nansum(np.array(technicals) * np.array(technical_weights))

    return fundamental_score, technical_score


# Combining scores
def combine_scores(fundamental_score, technical_score):
    combined_score = (fundamental_score + technical_score) / 2
    return combined_score


# Future price prediction based on combined score and trend confirmation
def predict_future_price(hist, combined_score, trend_confirmation, future_date):
    current_price = hist['Close'].iloc[-1]
    days_to_predict = (future_date - datetime.now()).days

    if trend_confirmation:
        trend_factor = 1 + (combined_score / 100)
    else:
        trend_factor = 1 - (combined_score / 100)

    future_price = current_price * (trend_factor ** (days_to_predict / 365))
    return future_price


@app.route('/stock', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        future_date_str = request.form['future_date']
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d")

        hist, info = fetch_data(ticker)
        fundamentals = calculate_fundamentals(info)
        hist, ma50, ma200, rsi, macd, signal, macd_histogram, stochastic_k, stochastic_d, momentum = calculate_technicals(
            hist)

        trend_confirmation = ma50.iloc[-1] > ma200.iloc[-1]

        technicals = [ma50.iloc[-1], ma200.iloc[-1], rsi.iloc[-1], macd.iloc[-1], macd_histogram.iloc[-1],
                      stochastic_k.iloc[-1], stochastic_d.iloc[-1], momentum.iloc[-1]]

        fundamental_score, technical_score = assign_weights(fundamentals, technicals)
        combined_score = combine_scores(fundamental_score, technical_score)

        future_price = predict_future_price(hist, combined_score, trend_confirmation, future_date)

        return render_template('index6.html', ticker=ticker, future_date=future_date_str,
                               future_price=f"${future_price:.2f}")

    return render_template('index6.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
