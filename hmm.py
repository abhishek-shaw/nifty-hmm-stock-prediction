import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import plotly.graph_objects as go

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_volatility(data, window=20):
    return data.rolling(window=window).std()

def calculate_ema(data, window=20):
    return data.ewm(span=window, adjust=False).mean()

def identify_trend_dow_dynamic(df, window=2, timeframe='W'):
    resampled_df = df.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    resampled_df['Higher_High'] = resampled_df['High'].rolling(window=window).apply(lambda x: x.iloc[-1] == x.max())
    resampled_df['Lower_Low'] = resampled_df['Low'].rolling(window=window).apply(lambda x: x.iloc[-1] == x.min())
    resampled_df['Higher_Low'] = resampled_df['Low'].rolling(window=window).apply(lambda x: x.iloc[-1] > x.iloc[0])
    resampled_df['Lower_High'] = resampled_df['High'].rolling(window=window).apply(lambda x: x.iloc[-1] < x.iloc[0])


    def get_trend(row):
        if row['Higher_High'] and row['Higher_Low']:
            return 'bullish'
        elif row['Lower_Low'] and row['Lower_High']:
            return 'bearish'
        else:
            return 'sideways'

    resampled_df['trend'] = resampled_df.apply(get_trend, axis=1)
    return resampled_df.reindex(df.index, method='ffill')['trend']

def identify_trend_dow(df, window=20):
    df['Higher_High'] = df['High'].rolling(window=window).apply(lambda x: x.iloc[-1] == x.max())
    df['Lower_Low'] = df['Low'].rolling(window=window).apply(lambda x: x.iloc[-1] == x.min())
    df['Higher_Low'] = df['Low'].rolling(window=window).apply(lambda x: x.iloc[-1] > x.iloc[0])
    df['Lower_High'] = df['High'].rolling(window=window).apply(lambda x: x.iloc[-1] < x.iloc[0])

    def get_trend(row):
        if row['Higher_High'] and row['Higher_Low']:
            return 'bullish'
        elif row['Lower_Low'] and row['Lower_High']:
            return 'bearish'
        else:
            return 'sideways'

    df['trend'] = df.apply(get_trend, axis=1)
    return df['trend']

def draw_trend_lines(df):
    trend_lines = []
    current_trend = None
    start_index = 0

    for i in range(len(df)):
        if df['trend'].iloc[i] != current_trend:
            if current_trend in ['bullish', 'bearish']:
                end_index = i - 1
                start_price = df['Close'].iloc[start_index]
                end_price = df['Close'].iloc[end_index]
                trend_lines.append({
                    'x0': df.index[start_index],
                    'x1': df.index[end_index],
                    'y0': start_price,
                    'y1': end_price,
                    'color': 'green' if current_trend == 'bullish' else 'red'
                })
            current_trend = df['trend'].iloc[i]
            start_index = i

    # Add the last trend line
    if current_trend in ['bullish', 'bearish']:
        end_index = len(df) - 1
        start_price = df['Close'].iloc[start_index]
        end_price = df['Close'].iloc[end_index]
        trend_lines.append({
            'x0': df.index[start_index],
            'x1': df.index[end_index],
            'y0': start_price,
            'y1': end_price,
            'color': 'green' if current_trend == 'bullish' else 'red'
        })

    return trend_lines

def backtest_model(model, X, y, le):
    predictions = model.predict(X)
    predictions = le.inverse_transform(predictions)
    actual = le.inverse_transform(y)

    correct = sum(predictions == actual)
    total = len(predictions)
    accuracy = correct / total

    hit_ratio = {
        'bullish': {'correct': 0, 'total': 0},
        'bearish': {'correct': 0, 'total': 0},
        'sideways': {'correct': 0, 'total': 0}
    }

    for pred, act in zip(predictions, actual):
        hit_ratio[act]['total'] += 1
        if pred == act:
            hit_ratio[act]['correct'] += 1

    for trend in hit_ratio:
        if hit_ratio[trend]['total'] > 0:
            hit_ratio[trend]['ratio'] = hit_ratio[trend]['correct'] / hit_ratio[trend]['total']
        else:
            hit_ratio[trend]['ratio'] = 0

    return accuracy, hit_ratio

# Calculate support and resistance levels
def calculate_support_resistance(data, window=20):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

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

# Load the data
df = pd.read_csv('nifty50.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Calculate technical indicators
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'], df['Signal'] = calculate_macd(df['Close'])
df['Volatility'] = calculate_volatility(df['Close'])
df['EMA'] = calculate_ema(df['Close'])

# Create features: percentage change and other indicators
df['pct_change'] = df['Close'].pct_change()
df['range'] = (df['High'] - df['Low']) / df['Close']

# Identify trend using Dow Theory on weekly timeframe
#df['trend'] = identify_trend_dow_dynamic(df)
df['trend'] = identify_trend_dow(df)

# Calculate support and Resistance
df = calculate_support_resistance(df)


# Drop rows with NaN values
df.dropna(inplace=True)

# Prepare features for HMM
features = ['pct_change', 'range', 'RSI', 'MACD', 'Volatility', 'EMA']
X = df[features].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use LabelEncoder for 'trend'
le = LabelEncoder()
y = le.fit_transform(df['trend'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Train a Hidden Markov Model
model = hmm.GaussianHMM(n_components=3, n_iter=1000, random_state=42)
model.fit(X_train)

# Predict the trend of the next day
last_features = X_scaled[-1].reshape(1, -1)
trend_label = model.predict(last_features)
trend = le.inverse_transform([trend_label[-1]])[0]

# Predict the next day's closing price using a simple moving average
next_close = df['Close'].rolling(window=5).mean().iloc[-1]

# Backtest the model
accuracy, hit_ratio = backtest_model(model, X_test, y_test, le)


# Filter the last 6 months of data
six_months_ago = df.index[-1] - pd.DateOffset(months=6)
df_last_6_months = df[df.index >= six_months_ago]

# Get automated swing highs and lows for the last 6 months
swing_highs, swing_lows = find_swing_points(df_last_6_months)


# Create candlestick chart with trend lines
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                             low=df['Low'], close=df['Close'], name='Price'))

# Resistance and Support lines
fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'],
                         mode='lines', name='Resistance', line=dict(color='red', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=df['Support'],
                         mode='lines', name='Support', line=dict(color='green', width=2, dash='dash')))

# Add EMA to candlestick chart
fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], name='EMA', line=dict(color='orange')))

# Add trend lines
trend_lines = draw_trend_lines(df)
for line in trend_lines:
    fig.add_shape(type="line", x0=line['x0'], y0=line['y0'], x1=line['x1'], y1=line['y1'],
                  line=dict(color=line['color'], width=2))

# Add predicted next day's close
fig.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], y=[next_close],
                         mode='markers+text', name='Predicted Next Close',
                         marker=dict(color='orange', size=10),
                         text=[f'Predicted Price: {next_close:.2f} <br>Predicted Trend: {trend}'],
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

# Update layout
fig.update_layout(
    title_text=f'Stock Price Prediction (Predicted Trend: {trend})',
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    yaxis=dict(range=[y_min, y_max]),
    xaxis=dict(range=[x_min, x_max])
)

# Add backtesting results to the chart
backtest_text = f"""
Backtesting Results:
Overall Accuracy: {accuracy:.2f}
Hit Ratios:
  Bullish: {hit_ratio['bullish']['ratio']:.2f} ({hit_ratio['bullish']['correct']}/{hit_ratio['bullish']['total']})
  Bearish: {hit_ratio['bearish']['ratio']:.2f} ({hit_ratio['bearish']['correct']}/{hit_ratio['bearish']['total']})
  Sideways: {hit_ratio['sideways']['ratio']:.2f} ({hit_ratio['sideways']['correct']}/{hit_ratio['sideways']['total']})
"""

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.02, y=0.98,
    text=backtest_text,
    showarrow=False,
    font=dict(size=10),
    align="left",
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    borderwidth=1
)


# Show the plot
fig.show()

# Save the figure to an HTML file
fig.write_html('chart.html')  # Output file name

print("Chart has been saved as 'chart.html'")

print(f"Predicted trend: {trend}")
print(f"Predicted next day's closing price: {next_close:.2f}")
print("The interactive chart has been saved as 'stock_prediction.html'")
print(backtest_text)
