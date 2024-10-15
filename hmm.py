import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('nifty50.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Calculate returns
df['Returns'] = df['Close'].pct_change()
df = df.dropna()

# Feature Engineering: Adding Technical Indicators
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate technical indicators
df['MA20'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
df['RSI'] = calculate_rsi(df['Close'], window=14)  # 14-day RSI
df['Volatility'] = df['Returns'].rolling(window=20).std()  # 20-day volatility

# Calculate MACD
ema_short = df['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
ema_long = df['Close'].ewm(span=26, adjust=False).mean()   # 26-day EMA
df['MACD'] = ema_short - ema_long  # MACD line
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line

# Prepare the data for HMM
features = ['Returns', 'MA20', 'RSI', 'Volatility', 'MACD', 'Signal']
X = df[features].dropna().values
y = df['Close'][df[features].dropna().index].values  # Use close prices as labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the HMM
n_states = 5
model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)

# Fit the model and predict hidden states
model.fit(X_train_scaled)
hidden_states = model.predict(X_train_scaled)

# Map hidden states to trend labels
trend_map = {
    0: 'Strong Bullish',
    1: 'Mild Bullish',
    2: 'Sideways',
    3: 'Mild Bearish',
    4: 'Strong Bearish'
}

# Assign trends only to the rows in the original DataFrame that were used for training
df.loc[df.index[:len(hidden_states)], 'Trend'] = [trend_map[state] for state in hidden_states]

# Calculate support and resistance levels
def calculate_support_resistance(data, window=20):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

df = calculate_support_resistance(df)

# Predict future trend, price, support, and resistance
last_features = [df['Returns'].iloc[-1], df['MA20'].iloc[-1], df['RSI'].iloc[-1], df['Volatility'].iloc[-1], df['MACD'].iloc[-1], df['Signal'].iloc[-1]]
last_price = df['Close'].iloc[-1]

def predict_next_price(model, last_price, last_features):
    # Transform the last features (Returns, MA20, RSI, Volatility, MACD, Signal)
    last_features_scaled = scaler.transform([last_features])  # Transform all features together
    next_state = model.predict(last_features_scaled)[0]

    # Instead of taking the first mean, we take the mean of the state which corresponds to our return feature
    next_return = model.means_[next_state][0]

    # Calculate the next price based on return
    next_price = last_price * (1 + next_return)

    return next_price

next_price = predict_next_price(model, last_price, last_features)

# Predict future support and resistance
recent_highs = df['High'][-20:].max()
recent_lows = df['Low'][-20:].min()
future_support = recent_lows
future_resistance = recent_highs

# Add future prediction to the DataFrame for plotting
future_date = df.index[-1] + pd.Timedelta(days=1)
future_row = pd.DataFrame({
    'Open': [np.nan],
    'High': [np.nan],
    'Low': [np.nan],
    'Close': [next_price],
    'Volume': [0],
    'Support': [future_support],
    'Resistance': [future_resistance],
    'Trend': [trend_map[model.predict(scaler.transform([last_features]))[0]]]
}, index=[future_date])

df = pd.concat([df, future_row])

# Function to find automated swing points based on defined thresholds
def find_swing_points(df, window=5):
    highs = df['High']
    lows = df['Low']
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        # Detect swing high
        if (highs[i] == highs[i-window:i+window+1].max() and
            (i == window or highs[i] > highs[i-1]) and
            (i == len(df)-window-1 or highs[i] > highs[i+1])):
            swing_highs.append((df.index[i], highs[i]))

        # Detect swing low
        if (lows[i] == lows[i-window:i+window+1].min() and
            (i == window or lows[i] < lows[i-1]) and
            (i == len(df)-window-1 or lows[i] < lows[i+1])):
            swing_lows.append((df.index[i], lows[i]))

    return swing_highs, swing_lows

# Filter the last 6 months of data
six_months_ago = df.index[-1] - pd.DateOffset(months=6)
df_last_6_months = df[df.index >= six_months_ago]

# Get automated swing highs and lows for the last 6 months
swing_highs, swing_lows = find_swing_points(df_last_6_months)

# Create the candlestick chart
fig = go.Figure()

# Candlestick trace
fig.add_trace(go.Candlestick(x=df.index,
                              open=df['Open'],
                              high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              name='Candlestick'))

# Resistance and Support lines
fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'],
                         mode='lines', name='Resistance', line=dict(color='red', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=df['Support'],
                         mode='lines', name='Support', line=dict(color='green', width=2, dash='dash')))

# Future price marker
fig.add_trace(go.Scatter(x=[future_date], y=[next_price],
                         mode='markers+text', name='Future Price',
                         marker=dict(color='orange', size=10),
                         text=[f'Future Price: {next_price:.2f} <br>Future Trend: {trend_map[model.predict(scaler.transform([last_features]))[0]]}'],
                         textposition='top center'))

# Refine trendlines based on the swing points
if swing_highs:
    swing_high_x = [point[0] for point in swing_highs]
    swing_high_y = [point[1] for point in swing_highs]

    # Add trendlines connecting the swing highs
    fig.add_trace(go.Scatter(x=swing_high_x, y=swing_high_y,
                             mode='lines', name='Trendline (High)',
                             line=dict(color='blue', width=2)))

if swing_lows:
    swing_low_x = [point[0] for point in swing_lows]
    swing_low_y = [point[1] for point in swing_lows]

    # Add trendlines connecting the swing lows
    fig.add_trace(go.Scatter(x=swing_low_x, y=swing_low_y,
                             mode='lines', name='Trendline (Low)',
                             line=dict(color='orange', width=2)))

# Set y-axis range based on the last 6 months of price data
y_min = df_last_6_months['Low'].min() * 0.95  # 5% below the lowest low
y_max = df_last_6_months['High'].max() * 1.05  # 5% above the highest high

# Set x-axis range to show the last 6 months + 5 days
x_min = df_last_6_months.index.min()
x_max = df_last_6_months.index.max() + pd.DateOffset(days=15)

fig.update_layout(title='Nifty50 Stock Price Prediction',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  yaxis=dict(range=[y_min, y_max]),
                  xaxis=dict(range=[x_min, x_max]))

# Show the plot
fig.show()

# Save the figure to an HTML file
fig.write_html('chart.html')  # Output file name

print("Chart has been saved as 'chart.html'")
