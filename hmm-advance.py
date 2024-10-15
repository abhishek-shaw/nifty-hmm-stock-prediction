import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Load and preprocess data (unchanged)
df = pd.read_csv('nifty50.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df['Returns'] = df['Close'].pct_change()
df = df.dropna()

# Feature Engineering (unchanged)
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['MA20'] = df['Close'].rolling(window=20).mean()
df['RSI'] = calculate_rsi(df['Close'], window=14)
df['Volatility'] = df['Returns'].rolling(window=20).std()
ema_short = df['Close'].ewm(span=12, adjust=False).mean()
ema_long = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_short - ema_long
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Prepare data for HMM (unchanged)
features = ['Returns', 'MA20', 'RSI', 'Volatility', 'MACD', 'Signal']
X = df[features].dropna().values
y = df['Close'][df[features].dropna().index].values

# Split data and scale features (unchanged)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# HMM Model (unchanged)
n_states = 5
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=2000, random_state=42)
model.fit(X_train_scaled)

# Predict hidden states for all data
hidden_states = model.predict(scaler.transform(X))

# Map hidden states to trend labels
trend_map = {
    0: 'Strong Bullish',
    1: 'Mild Bullish',
    2: 'Sideways',
    3: 'Mild Bearish',
    4: 'Strong Bearish'
}

# Assign trends to the DataFrame
# df['Trend'] = [trend_map[state] for state in hidden_states]

# Prepare data for HMM (drop rows with NaN in features)
df_cleaned = df.dropna(subset=features)

# Create the feature matrix for HMM prediction
X_cleaned = df_cleaned[features].values

# Predict hidden states for the cleaned data
hidden_states = model.predict(scaler.transform(X_cleaned))

# Assign trends to the cleaned DataFrame
df_cleaned['Trend'] = [trend_map[state] for state in hidden_states]

# Now merge or fill missing data back into the original DataFrame
df['Trend'] = df_cleaned['Trend']


# Improved price prediction function
def predict_next_price(model, last_price, last_features, n_days=30):
    predictions = []
    states = []
    current_features = last_features.copy()

    for _ in range(n_days):
        current_features_scaled = scaler.transform([current_features])
        next_state = model.predict(current_features_scaled)[0]
        next_return = model.means_[next_state][0]
        next_price = last_price * (1 + next_return)

        predictions.append(next_price)
        states.append(next_state)

        # Update features for next prediction
        current_features[0] = next_return
        current_features[1] = (current_features[1] * 19 + next_price) / 20  # Update MA20
        # Simplistic updates for other features
        current_features[2] = max(0, min(100, current_features[2] + np.random.normal(0, 1)))  # RSI
        current_features[3] = max(0, current_features[3] + np.random.normal(0, 0.001))  # Volatility
        current_features[4] += np.random.normal(0, 0.1)  # MACD
        current_features[5] += np.random.normal(0, 0.1)  # Signal

        last_price = next_price

    return predictions, states

# Predict future prices
last_features = [df['Returns'].iloc[-1], df['MA20'].iloc[-1], df['RSI'].iloc[-1], df['Volatility'].iloc[-1], df['MACD'].iloc[-1], df['Signal'].iloc[-1]]
last_price = df['Close'].iloc[-1]
future_prices, future_states = predict_next_price(model, last_price, last_features, n_days=30)

# Add future predictions to DataFrame
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame({
    'Close': future_prices,
    'Trend': [trend_map[state] for state in future_states]
}, index=future_dates)

df = pd.concat([df, future_df])

# Calculate Support and Resistance levels
def calculate_support_resistance(data, window=20):
    support = data['Low'].rolling(window=window).min()
    resistance = data['High'].rolling(window=window).max()
    return support, resistance

support, resistance = calculate_support_resistance(df)

# Create the plot
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))

# Support and Resistance lines
fig.add_trace(go.Scatter(x=df.index, y=support, mode='lines', name='Support', line=dict(color='green', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=resistance, mode='lines', name='Resistance', line=dict(color='red', width=2, dash='dash')))

# Future price predictions
fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='markers+lines',
                         name='Future Prices', line=dict(color='purple', dash='dot'),
                         marker=dict(size=8, symbol='star')))

# Hidden States
colors = {'Strong Bullish': 'darkgreen', 'Mild Bullish': 'lightgreen',
          'Sideways': 'yellow', 'Mild Bearish': 'orange', 'Strong Bearish': 'red'}

for trend in trend_map.values():
    mask = df['Trend'] == trend
    fig.add_trace(go.Scatter(x=df.index[mask], y=df['Close'][mask], mode='markers',
                             name=f'State: {trend}', marker=dict(size=6, color=colors[trend]),
                             showlegend=True))

# Layout settings
fig.update_layout(title='Nifty50 Stock Price Prediction with Hidden States and Extended Forecast',
                  xaxis_title='Date', yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

# Adjust y-axis range to show full support and resistance
y_min = min(df['Low'].min(), support.min()) * 0.99
y_max = max(df['High'].max(), resistance.max()) * 1.01
fig.update_yaxes(range=[y_min, y_max])

# Show the plot
fig.show()

# Save the figure
fig.write_html('nifty50_prediction_with_hidden_states_and_forecast.html')

print("Chart with hidden states and extended forecast has been saved as 'nifty50_prediction_with_hidden_states_and_forecast.html'")
