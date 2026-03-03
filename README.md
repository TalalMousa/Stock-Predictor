# Stock Predictor

A Streamlit app that predicts the next trading day's stock price for a small set of supported stocks using pre-trained PyTorch LSTM models.

## Features

- Select a supported stock symbol from the UI.
- Download historical market data from Yahoo Finance with `yfinance`.
- Load a pre-trained LSTM model and its saved `MinMaxScaler`.
- Generate a next-day price prediction.
- Visualize recent price action with:
  - actual closing prices
  - 20-day simple moving average (SMA)
  - 20-day exponential moving average (EMA)
  - next-day prediction marker

## Supported Stocks

The current app includes trained models for:

- `AAPL`
- `MSFT`
- `TSLA`
- `AMZN`

## Tech Stack

- Python
- Streamlit
- PyTorch
- scikit-learn
- pandas
- numpy
- yfinance
- Plotly

## Project Structure

```text
Stock_Predictor/
|-- app.py
|-- requirements.txt
|-- models/
|   |-- AAPL_model.pt
|   |-- AMZN_model.pt
|   |-- MSFT_model.pt
|   `-- TSLA_model.pt
`-- Stock_Predictor_LSTM (1).ipynb
```

## Installation

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## How It Works

1. The app loads a pre-trained LSTM model for the selected stock.
2. It downloads historical price data starting from `2015-01-01`.
3. It extracts the closing price series and scales it using the saved scaler.
4. It uses the latest 60 time steps as input to the model.
5. It predicts the next day's closing price and converts it back to the original scale.

## Notes

- The app depends on Yahoo Finance data availability.
- Predictions are based on pre-trained models already stored in the repository.
- This project is for educational and experimental use, not financial advice.
- `requirements.txt` currently includes both `scikit-learn` and `sklearn`; in most cases, `scikit-learn` is the package you need.

