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

def identify_trend_dow_improved(df, window=5, range_threshold=0.02):
    """
    Enhanced Dow Theory trend identification with range-bound market detection
    """
    # Calculate rolling max and min for comparison
    df['Rolling_High'] = df['High'].rolling(window=window).max()
    df['Rolling_Low'] = df['Low'].rolling(window=window).min()

    # Calculate price range as percentage of average price
    df['Price_Range'] = (df['Rolling_High'] - df['Rolling_Low']) / df['Close'].rolling(window=window).mean()

    # Calculate rate of change for highs and lows
    df['High_ROC'] = df['Rolling_High'].pct_change(periods=window)
    df['Low_ROC'] = df['Rolling_Low'].pct_change(periods=window)

    # Initialize trend components
    df['Higher_High'] = False
    df['Lower_Low'] = False
    df['Higher_Low'] = False
    df['Lower_High'] = False
    df['Range_Bound'] = False

    for i in range(window, len(df)):
        # Get previous window's data
        prev_window = df.iloc[i-window:i]
        current_price = df.iloc[i]

        # Calculate range-bound characteristics
        price_range = df.loc[df.index[i], 'Price_Range']
        high_roc = abs(df.loc[df.index[i], 'High_ROC'])
        low_roc = abs(df.loc[df.index[i], 'Low_ROC'])

        # Detect range-bound market
        df.loc[df.index[i], 'Range_Bound'] = (
            price_range < range_threshold and
            high_roc < range_threshold and
            low_roc < range_threshold
        )

        # Higher High: Current high is higher than previous window's high
        df.loc[df.index[i], 'Higher_High'] = (
            current_price['High'] > prev_window['High'].max() and
            current_price['Close'] > prev_window['Close'].mean()
        )

        # Lower Low: Current low is lower than previous window's low
        df.loc[df.index[i], 'Lower_Low'] = (
            current_price['Low'] < prev_window['Low'].min() and
            current_price['Close'] < prev_window['Close'].mean()
        )

        # Higher Low: Current low is higher than previous window's low
        df.loc[df.index[i], 'Higher_Low'] = (
            current_price['Low'] > prev_window['Low'].min() and
            current_price['Close'] > prev_window['Close'].min()
        )

        # Lower High: Current high is lower than previous window's high
        df.loc[df.index[i], 'Lower_High'] = (
            current_price['High'] < prev_window['High'].max() and
            current_price['Close'] < prev_window['Close'].max()
        )

    def determine_trend(row):
        """Determine trend based on price action patterns including range-bound detection"""
        if row['Range_Bound']:
            return 'sideways'
        elif row['Higher_High'] and row['Higher_Low']:
            return 'bullish'
        elif row['Lower_Low'] and row['Lower_High']:
            return 'bearish'
        elif row['Higher_High'] and not row['Lower_Low']:
            return 'bullish'  # Potential bullish breakout
        elif row['Lower_Low'] and not row['Higher_High']:
            return 'bearish'  # Potential bearish breakdown
        else:
            return 'sideways'

    df['trend'] = df.apply(determine_trend, axis=1)

    # Add range boundaries for sideways trend
    df['Range_High'] = None
    df['Range_Low'] = None

    for i in range(window, len(df)):
        if df.iloc[i]['trend'] == 'sideways':
            df.loc[df.index[i], 'Range_High'] = df['High'].iloc[i-window:i].max()
            df.loc[df.index[i], 'Range_Low'] = df['Low'].iloc[i-window:i].min()

    return df['trend'], df['Range_High'], df['Range_Low']

def predict_price_ranges(df, volatility_window=10, atr_window=14):
    """
    Enhanced price range prediction with range-bound market considerations
    """
    # Calculate True Range
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )

    # Calculate ATR with shorter window
    atr = df['TR'].rolling(window=atr_window).mean().iloc[-1]

    # Calculate recent ranges
    recent_ranges = (df['High'] - df['Low']).rolling(window=volatility_window).mean().iloc[-1]

    # Get range boundaries if in sideways trend
    range_high = df['Range_High'].iloc[-1] if pd.notna(df['Range_High'].iloc[-1]) else None
    range_low = df['Range_Low'].iloc[-1] if pd.notna(df['Range_Low'].iloc[-1]) else None

    # Calculate trend-based factors
    if trend == 'sideways':
        # Use range boundaries for sideways market
        if range_high is not None and range_low is not None:
            range_middle = (range_high + range_low) / 2
            range_size = (range_high - range_low) * 0.3  # 30% of range size
            predicted_high = min(range_high, next_close + range_size)
            predicted_low = max(range_low, next_close - range_size)
        else:
            range_size = recent_ranges * 0.5  # More conservative range for sideways
            predicted_high = next_close + (range_size / 2)
            predicted_low = next_close - (range_size / 2)
    else:
        # Calculate trend-based ranges
        trend_factor = 1.2 if trend == 'bullish' else 0.8 if trend == 'bearish' else 1.0
        avg_high_close_range = (df['High'] - df['Close']).rolling(window=volatility_window).mean().iloc[-1]
        avg_close_low_range = (df['Close'] - df['Low']).rolling(window=volatility_window).mean().iloc[-1]

        predicted_high = next_close + (avg_high_close_range * trend_factor)
        predicted_low = next_close - (avg_close_low_range * trend_factor)

    # Apply volatility-based constraints
    max_range = recent_ranges * 1.2
    current_range = predicted_high - predicted_low

    if current_range > max_range:
        center = next_close
        predicted_high = center + (max_range / 2)
        predicted_low = center - (max_range / 2)

    # Apply maximum daily movement constraints
    last_high = df['High'].iloc[-1]
    last_low = df['Low'].iloc[-1]

    max_up_move = last_high * 0.015  # 1.5% maximum up move
    max_down_move = last_low * 0.015  # 1.5% maximum down move

    predicted_high = min(last_high + max_up_move, predicted_high)
    predicted_low = max(last_low - max_down_move, predicted_low)

    return predicted_high, predicted_low, range_high, range_low

def draw_trend_lines_improved(df):
    """
    Improved trend line drawing with better support for price action
    """
    trend_lines = []
    swing_points = []
    current_trend = None
    start_index = 0

    # Identify significant swing points
    for i in range(1, len(df)-1):
        if df['trend'].iloc[i] != current_trend:
            swing_points.append(i)
            current_trend = df['trend'].iloc[i]

    # Draw trend lines between significant points
    for i in range(len(swing_points)-1):
        start_idx = swing_points[i]
        end_idx = swing_points[i+1]

        trend = df['trend'].iloc[start_idx:end_idx].mode()[0]

        if trend in ['bullish', 'bearish']:
            # Use OHLC data to determine line points
            if trend == 'bullish':
                start_price = df['Low'].iloc[start_idx]
                end_price = df['High'].iloc[end_idx]
            else:
                start_price = df['High'].iloc[start_idx]
                end_price = df['Low'].iloc[end_idx]

            trend_lines.append({
                'x0': df.index[start_idx],
                'x1': df.index[end_idx],
                'y0': start_price,
                'y1': end_price,
                'color': 'green' if trend == 'bullish' else 'red'
            })

    return trend_lines

def plot_price_action_markers(fig, df):
    """
    Add price action markers to the chart
    """
    # Plot Higher Highs
    hh_indices = df[df['Higher_High']].index
    fig.add_trace(go.Scatter(
        x=hh_indices,
        y=df.loc[hh_indices, 'High'],
        mode='markers',
        name='Higher Highs',
        marker=dict(
            symbol='triangle-up',
            size=10,
            color='green',
            line=dict(width=2)
        )
    ))

    # Plot Lower Lows
    ll_indices = df[df['Lower_Low']].index
    fig.add_trace(go.Scatter(
        x=ll_indices,
        y=df.loc[ll_indices, 'Low'],
        mode='markers',
        name='Lower Lows',
        marker=dict(
            symbol='triangle-down',
            size=10,
            color='red',
            line=dict(width=2)
        )
    ))

    return fig

def calculate_support_resistance(data, window=20):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

def find_swing_points(df, window=5):
    highs = df['High']
    lows = df['Low']
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - 1):  # Adjust the range to len(df) - 1
        if (highs.iloc[i] == highs.iloc[i-window:i+window+1].max() and
            (i == window or highs.iloc[i] > highs.iloc[i-1]) and
            (i == len(df)-window-1 or highs.iloc[i] > highs.iloc[i+1])):
            swing_highs.append((df.index[i], highs.iloc[i]))
    
        if (lows.iloc[i] == lows.iloc[i-window:i+window+1].min() and
            (i == window or lows.iloc[i] < lows.iloc[i-1]) and
            (i == len(df)-window-1 or lows.iloc[i] < lows.iloc[i+1])):
            swing_lows.append((df.index[i], lows.iloc[i]))
    
    return swing_highs, swing_lows

def backtest_model(model, X, y, le):
    """
    Modified backtest function to handle HMM predictions properly
    """
    # Get raw predictions
    predictions = model.predict(X)

    # Initialize counters for each possible state
    n_states = model.n_components
    state_mapping = {}

    # Count occurrences of each state for each trend
    state_trend_counts = {i: {} for i in range(n_states)}
    for pred, actual in zip(predictions, y):
        if actual not in state_trend_counts[pred]:
            state_trend_counts[pred][actual] = 0
        state_trend_counts[pred][actual] += 1

    # Map each state to the most common trend
    for state in range(n_states):
        if state_trend_counts[state]:
            most_common_trend = max(state_trend_counts[state].items(),
                                  key=lambda x: x[1])[0]
            state_mapping[state] = most_common_trend
        else:
            state_mapping[state] = 0  # Default mapping if state never occurred

    # Map predictions to trends using the state mapping
    mapped_predictions = np.array([state_mapping[pred] for pred in predictions])

    # Calculate accuracy
    correct = sum(mapped_predictions == y)
    total = len(mapped_predictions)
    accuracy = correct / total if total > 0 else 0

    # Transform numeric labels back to string labels
    predicted_trends = le.inverse_transform(mapped_predictions)
    actual_trends = le.inverse_transform(y)

    # Calculate hit ratio for each trend
    hit_ratio = {
        'bullish': {'correct': 0, 'total': 0},
        'bearish': {'correct': 0, 'total': 0},
        'sideways': {'correct': 0, 'total': 0}
    }

    for pred, act in zip(predicted_trends, actual_trends):
        hit_ratio[act]['total'] += 1
        if pred == act:
            hit_ratio[act]['correct'] += 1

    # Calculate ratios
    for trend in hit_ratio:
        if hit_ratio[trend]['total'] > 0:
            hit_ratio[trend]['ratio'] = hit_ratio[trend]['correct'] / hit_ratio[trend]['total']
        else:
            hit_ratio[trend]['ratio'] = 0

    return accuracy, hit_ratio

# Load the data
df = pd.read_csv('nifty50.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Calculate returns
df['Returns'] = df['Close'].pct_change()
df = df.dropna()


# Calculate technical indicators
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'], df['Signal'] = calculate_macd(df['Close'])
df['Volatility'] = calculate_volatility(df['Close'])
df['EMA'] = calculate_ema(df['Close'])
df['pct_change'] = df['Close'].pct_change()
df['range'] = (df['High'] - df['Low']) / df['Close']

# Identify trends and range boundaries
df['trend'], df['Range_High'], df['Range_Low'] = identify_trend_dow_improved(df)

print(f"Original dataset size: {len(df)}")
df = df.dropna()
print(f"Dataset size after removing NaN values: {len(df)}")

# Prepare features for HMM
features = ['pct_change', 'range', 'RSI', 'MACD', 'Volatility', 'EMA']
X = df[features].values

# Verify there are no NaN values
if np.isnan(X).any():
    print("Warning: NaN values found in features. Removing rows with NaN values...")
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    df = df[mask]
    print(f"Remaining samples after removing NaN: {len(X)}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare labels
le = LabelEncoder()
y = le.fit_transform(df['trend'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Train HMM
model = hmm.GaussianHMM(n_components=3, n_iter=1000, random_state=42)
model.fit(X_train)


# Also modify the prediction part after model training:
# Get model prediction
last_features = X_scaled[-1].reshape(1, -1)
state_prediction = model.predict(last_features)[0]

# Map the state to trend (use the same mapping logic as in backtest)
state_trend_counts = {i: {} for i in range(model.n_components)}
all_predictions = model.predict(X_scaled)
for pred, actual in zip(all_predictions, y):
    if actual not in state_trend_counts[pred]:
        state_trend_counts[pred][actual] = 0
    state_trend_counts[pred][actual] += 1

state_mapping = {}
for state in range(model.n_components):
    if state_trend_counts[state]:
        most_common_trend = max(state_trend_counts[state].items(),
                              key=lambda x: x[1])[0]
        state_mapping[state] = most_common_trend
    else:
        state_mapping[state] = 0

trend_label = state_mapping[state_prediction]
trend = le.inverse_transform([trend_label])[0]

# Predict next day's close (using 5-day moving average)
next_close = df['Close'].rolling(window=5).mean().iloc[-1]

# Predict price ranges
predicted_high, predicted_low, range_high, range_low = predict_price_ranges(df)

# Backtest the model
accuracy, hit_ratio = backtest_model(model, X_test, y_test, le)

# Create visualization
# Filter last 6 months of data for display
six_months_ago = df.index[-1] - pd.DateOffset(months=6)
df_last_6_months = df[df.index >= six_months_ago]

# Get swing points
swing_highs, swing_lows = find_swing_points(df_last_6_months)

# Create the base candlestick chart
fig = go.Figure()

# Add candlestick chart
fig.add_trace(go.Candlestick(
    x=df_last_6_months.index,
    open=df_last_6_months['Open'],
    high=df_last_6_months['High'],
    low=df_last_6_months['Low'],
    close=df_last_6_months['Close'],
    name='Price'
))

# Add EMA
fig.add_trace(go.Scatter(
    x=df_last_6_months.index,
    y=df_last_6_months['EMA'],
    name='EMA',
    line=dict(color='orange')
))

# Add next day predictions
next_day = df.index[-1] + pd.Timedelta(days=1)

# Add predicted range box
fig.add_shape(
    type="rect",
    x0=next_day,
    x1=next_day + pd.Timedelta(days=1),
    y0=predicted_low,
    y1=predicted_high,
    fillcolor="rgba(150,150,150,0.1)",
    line=dict(color="gray", width=1, dash="dot"),
    name="Predicted Range"
)

# Add predicted points
fig.add_trace(go.Scatter(
    x=[next_day],
    y=[predicted_high],
    mode='markers+text',
    name='Predicted High',
    marker=dict(color='green', size=6, symbol='triangle-up'),
    text=[f'H: {predicted_high:.2f}'],
    textposition='top center'
))

fig.add_trace(go.Scatter(
    x=[next_day],
    y=[predicted_low],
    mode='markers+text',
    name='Predicted Low',
    marker=dict(color='red', size=6, symbol='triangle-down'),
    text=[f'L: {predicted_low:.2f}'],
    textposition='bottom center'
))

fig.add_trace(go.Scatter(
    x=[next_day],
    y=[next_close],
    mode='markers+text',
    name='Predicted Close',
    marker=dict(color='blue', size=8, symbol='circle'),
    text=[f'C: {next_close:.2f}'],
    textposition='middle right'
))

# Add range boundaries for sideways trend
if trend == 'sideways' and range_high is not None and range_low is not None:
    fig.add_shape(
        type="rect",
        x0=df_last_6_months.index[-20],  # Last 20 days
        x1=next_day + pd.Timedelta(days=1),
        y0=range_low,
        y1=range_high,
        fillcolor="rgba(200,200,200,0.1)",
        line=dict(color="purple", width=1, dash="dot"),
        name="Range Boundaries"
    )

    # Add range boundary lines
    fig.add_trace(go.Scatter(
        x=[df.index[-1], next_day],
        y=[range_high, range_high],
        mode='lines',
        name='Range High',
        line=dict(color='purple', width=1, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=[df.index[-1], next_day],
        y=[range_low, range_low],
        mode='lines',
        name='Range Low',
        line=dict(color='purple', width=1, dash='dot')
    ))

# Prepare backtest text
backtest_text = f"""
Backtesting Results:
Overall Accuracy: {accuracy:.2f}
Hit Ratios:
Bullish: {hit_ratio['bullish']['ratio']:.2f} ({hit_ratio['bullish']['correct']}/{hit_ratio['bullish']['total']})
Bearish: {hit_ratio['bearish']['ratio']:.2f} ({hit_ratio['bearish']['correct']}/{hit_ratio['bearish']['total']})
Sideways: {hit_ratio['sideways']['ratio']:.2f} ({hit_ratio['sideways']['correct']}/{hit_ratio['sideways']['total']})

Predicted Ranges for Next Day:
High: {predicted_high:.2f}
Close: {next_close:.2f}
Low: {predicted_low:.2f}
Range: {(predicted_high - predicted_low):.2f}
"""

if trend == 'sideways' and range_high is not None:
    backtest_text += f"""
Range Boundaries:
High: {range_high:.2f}
Low: {range_low:.2f}
Range Size: {(range_high - range_low):.2f}
"""

# Set y-axis range
y_min = min(df_last_6_months['Low'].min() * 0.998, predicted_low * 0.998)
y_max = max(df_last_6_months['High'].max() * 1.002, predicted_high * 1.002)

# Update layout
fig.update_layout(
    title_text=f'Stock Price Prediction with Ranges (Predicted Trend: {trend})',
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    yaxis=dict(range=[y_min, y_max]),
    xaxis=dict(range=[six_months_ago, next_day + pd.Timedelta(days=2)]),
    showlegend=True
)

# Add backtest results annotation
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

# Save the chart
fig.write_html('chart.html')

# Print results
print(f"\nMarket Condition: {trend}")
print(f"Predicted High: {predicted_high:.2f}")
print(f"Predicted Close: {next_close:.2f}")
print(f"Predicted Low: {predicted_low:.2f}")
if trend == 'sideways' and range_high is not None:
    print(f"Range High: {range_high:.2f}")
    print(f"Range Low: {range_low:.2f}")
print(f"\nBacktesting Accuracy: {accuracy:.2f}")
print("Chart saved as 'chat.html'")
