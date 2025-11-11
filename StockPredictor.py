import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title(" Stock Predictor with Polynomial Forecast")
st.write("_Enter a ticker symbol to see historical prices and a polynomial regression forecast._")
st.caption("⚠️ Purely mathematical — not financial advice.")

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Ticker Symbol", "AAPL").strip().upper()
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365 * 5))
    end_date = st.date_input("End Date", datetime.now())
    degree = st.slider("Polynomial Degree", 1, 5, 2)
    prediction_years = st.slider("Forecast Horizon (Years)", 1, 10, 3)

if not ticker_input:
    st.info("Enter a ticker to begin.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ---------------------- Fetch Data ----------------------
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

with st.spinner(f"Fetching data for {ticker_input}..."):
    df = fetch_data(ticker_input, start_date, end_date)

if df is None or df.empty:
    st.error("No data found. Check the ticker or date range.")
    st.stop()

# ---------------------- Column Handling ----------------------
columns = list(df.columns)

# Decide if OHLC data exists
if all(col in columns for col in ['Open', 'High', 'Low', 'Close']):
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    price_col = 'Close'
    has_ohlc = True
else:
    has_ohlc = False
    # Use adjusted close or close if available
    if 'Adj Close' in columns:
        price_col = 'Adj Close'
    elif 'Close' in columns:
        price_col = 'Close'
    else:
        # Fall back to first numeric column if nothing else
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            st.error("No numeric columns found in data.")
            st.stop()
        price_col = num_cols[0]

# ---------------------- Regression ----------------------
x = np.array([d.toordinal() for d in df.index])
y = df[price_col].to_numpy().flatten()

try:
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
except Exception as e:
    st.error(f"Regression failed: {e}")
    st.stop()

fitted = poly(x)
last_date = df.index[-1]
future_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    end=last_date + pd.DateOffset(years=prediction_years),
    freq='B'
)
x_future = np.array([d.toordinal() for d in future_dates])
y_future = poly(x_future)

# ---------------------- Plot ----------------------
fig = go.Figure()

if has_ohlc:
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))
else:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name='Historical Price',
        line=dict(color='royalblue', width=2),
        hovertemplate='Date: %{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>'
    ))

fig.add_trace(go.Scatter(
    x=df.index,
    y=fitted,
    mode='lines',
    name=f'Fitted (deg {degree})',
    line=dict(dash='dot', color='orange', width=2),
    hovertemplate='Date: %{x|%b %d, %Y}<br>Fitted: $%{y:.2f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=future_dates,
    y=y_future,
    mode='lines',
    name=f'Forecast ({prediction_years}y)',
    line=dict(dash='dash', color='lime', width=2),
    hovertemplate='Date: %{x|%b %d, %Y}<br>Forecast: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title=f"{ticker_input} – Historical, Fitted & Forecasted Prices",
    xaxis_title="Date",
    yaxis_title="Stock Price (USD)",
    hovermode='x unified',
    template='plotly_dark',
    height=650,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------- Data Preview ----------------------
st.subheader("Data Preview")
st.dataframe(df.tail(10))

st.subheader("Polynomial Coefficients")
coeff_df = pd.DataFrame({
    "Term": [f"x^{i}" for i in range(degree, -1, -1)],
    "Coefficient": [f"{c:.2e}" for c in coeffs]
})
st.table(coeff_df)

st.caption("⚠️ Polynomial regression is purely mathematical and not a reliable predictor of stock prices.")
