# Cointegration Trading Strategy

This repository implements a Cointegration Trading Strategy using Python. The strategy involves identifying cointegrated stock pairs and performing statistical arbitrage by trading the spread between them. The example in this code uses FedEx (FDX) and UPS (UPS).

## Features

- **Data Fetching**: Pulls historical data for the specified stock pairs from Polygon.io.
- **Cointegration Check**: Uses OLS regression and the Augmented Dickey-Fuller (ADF) test to check for cointegration between the two stocks.
- **Backtesting**: Simulates trades based on the spread between the cointegrated pairs using a Bollinger Bands strategy.
- **Statistical Analysis**: Calculates the win rate, t-statistic, and p-value to evaluate the strategy's performance.
- **Visualization**: Plots various performance metrics including cumulative PnL, spread, and the performance of individual stocks.

## Requirements

- `requests`
- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `pandas_market_calendars`
- `scipy`

You can install the required Python libraries using pip:

```bash
pip install requests pandas numpy matplotlib statsmodels pandas_market_calendars scipy
