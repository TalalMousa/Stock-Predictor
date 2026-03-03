import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = "models"


# PyTorch Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        return self.fc(out)

from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals({MinMaxScaler})


# Load PyTorch Model (.pt)
def load_pytorch_model(symbol):
    model_path = os.path.join(MODELS_DIR, f"{symbol}_model.pt")

    if not os.path.exists(model_path):
        st.error(f"MODEL NOT FOUND: {model_path}")
        st.stop()

    # Allow sklearn MinMaxScaler to be unpickled
    from sklearn.preprocessing import MinMaxScaler
    import torch.serialization
    torch.serialization.add_safe_globals({MinMaxScaler})

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = LSTMModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler = checkpoint["scaler"]
    return model, scaler


# Predict Next-Day Price
def predict_next_day(df, model, scaler):
    prices = df["close_price"].values.reshape(-1, 1)
    scaled = scaler.transform(prices)
    seq = torch.tensor(scaled[-60:], dtype=torch.float32).unsqueeze(0)
    pred_scaled = model(seq).detach().numpy()
    return scaler.inverse_transform(pred_scaled)[0][0]


# Streamlit UI
st.title("Stock Price Prediction")
st.write("Choose a stock and get next-day predictions using your trained LSTM model.")

symbols = ["AAPL", "MSFT", "TSLA", "AMZN"]
symbol = st.selectbox("Select a stock:", symbols)

# Load model
model, scaler = load_pytorch_model(symbol)

# Download real stock data
df = yf.download(symbol, start="2015-01-01")

if df.empty:
    st.error("Yahoo Finance returned no data!")
    st.stop()

# Normalize close column name
if "Close" in df.columns:
    df["close_price"] = df["Close"]
elif "Adj Close" in df.columns:
    df["close_price"] = df["Adj Close"]
else:
    st.error("No Close or Adj Close data available!")
    st.stop()

# FIX: Ensure datetime index
df.index = pd.to_datetime(df.index)

st.write("RAW DATA SAMPLE:", df.tail())

# Predict
prediction = predict_next_day(df, model, scaler)

st.subheader(f"Next-Day Prediction for {symbol}")
st.success(f"Predicted Price: **${prediction:.2f}**")


# Plot price + indicators
df_last = df.tail(400).copy()

df_last["SMA_20"] = df_last["close_price"].rolling(20).mean()
df_last["EMA_20"] = df_last["close_price"].ewm(span=20, adjust=False).mean()

fig = go.Figure()

# Actual Price
fig.add_trace(go.Scatter(
    x=df_last.index,
    y=df_last["close_price"],
    name="Actual Price",
    mode="lines",
    line=dict(color="blue")
))

# SMA
fig.add_trace(go.Scatter(
    x=df_last.index,
    y=df_last["SMA_20"],
    name="SMA 20",
    mode="lines",
    line=dict(color="orange")
))

# EMA
fig.add_trace(go.Scatter(
    x=df_last.index,
    y=df_last["EMA_20"],
    name="EMA 20",
    mode="lines",
    line=dict(color="green")
))

# Prediction marker
fig.add_trace(go.Scatter(
    x=[df_last.index[-1] + pd.Timedelta(days=1)],
    y=[prediction],
    mode="markers",
    name="Next-Day Prediction",
    marker=dict(size=12, color="red")
))

fig.update_layout(
    title=f"{symbol} Price Chart (Last 400 Days)",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    height=500,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

