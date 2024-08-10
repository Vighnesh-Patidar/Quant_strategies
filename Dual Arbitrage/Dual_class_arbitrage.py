import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from pandas_market_calendars import get_calendar
from scipy.stats import ttest_ind

class CointegrationStrategy:
    def __init__(self, tickers_1, tickers_2, api_key, start_date="2015-01-01"):
        self.tickers_1 = tickers_1
        self.tickers_2 = tickers_2
        self.api_key = api_key
        self.start_date = start_date
        self.calendar = get_calendar("NYSE")
        self.trading_dates = self.calendar.schedule(
            start_date=self.start_date,
            end_date=datetime.today()
        ).index.strftime("%Y-%m-%d").values
        self.portfolio_1_data = None
        self.portfolio_2_data = None
        self.combined_data = None
        self.trades_df = None

    def fetch_data(self, tickers):
        data_list = []
        for stock in tickers:
            stock_data = pd.json_normalize(requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/{stock}/range/1/day/{self.trading_dates[0]}/{self.trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}").json()[
                                           "results"]).set_index("t")
            stock_data.index = pd.to_datetime(stock_data.index, unit="ms", utc=True).tz_convert("America/New_York")
            stock_data["pct_change"] = stock_data["c"].pct_change()
            stock_data["ticker"] = stock
            stock_data = stock_data[["c", "pct_change", "ticker"]].dropna()
            data_list.append(stock_data)
        return pd.concat(data_list)

    def preprocess_data(self):
        self.portfolio_1_data = self.fetch_data(self.tickers_1).groupby(level=0).mean(numeric_only=True)
        self.portfolio_1_data["portfolio_performance"] = self.portfolio_1_data["pct_change"].cumsum() * 100

        self.portfolio_2_data = self.fetch_data(self.tickers_2).groupby(level=0).mean(numeric_only=True)
        self.portfolio_2_data["portfolio_performance"] = self.portfolio_2_data["pct_change"].cumsum() * 100

        self.combined_data = pd.concat([self.portfolio_1_data.add_prefix("1_"), self.portfolio_2_data.add_prefix("2_")],
                                       axis=1).dropna()
        self.combined_data["spread"] = abs(
            self.combined_data["1_portfolio_performance"] - self.combined_data["2_portfolio_performance"])

    def check_cointegration(self):
        x = sm.add_constant(self.combined_data['1_portfolio_performance'].values)
        y = self.combined_data['2_portfolio_performance'].values
        result = sm.OLS(y, x).fit()
        residual = result.resid
        adf_result = adfuller(residual)
        return adf_result[1], result.rsquared  # Return p-value and R-squared

    def backtest(self):
        self.combined_data['spread_ma'] = self.combined_data['spread'].rolling(window=200).mean()
        self.combined_data['std_dev'] = self.combined_data['spread'].rolling(window=200).std()
        self.combined_data['upper_band'] = self.combined_data['spread_ma'] + (self.combined_data['std_dev'] * 1)
        self.combined_data['lower_band'] = self.combined_data['spread_ma'] - (self.combined_data['std_dev'] * 1)
        self.combined_data["trading_signal"] = self.combined_data.apply(
            lambda row: 1 if (row['spread'] > row['upper_band']) else 0, axis=1)
        self.combined_data['entry_signal'] = (self.combined_data['trading_signal'] == 1) & (
                    self.combined_data['trading_signal'].shift(1) != 1)
        self.combined_data['exit_signal'] = (self.combined_data['trading_signal'] == 0) & (
                    self.combined_data['trading_signal'].shift(1) != 0)
        self.combined_data["underperformer_price"] = self.combined_data.apply(
            lambda row: row['1_c'] if (row['2_portfolio_performance'] > row['1_portfolio_performance']) else row['2_c'],
            axis=1)
        self.combined_data["overperformer_price"] = self.combined_data.apply(
            lambda row: row['1_c'] if (row['2_portfolio_performance'] < row['1_portfolio_performance']) else row['2_c'],
            axis=1)

        trades = []
        day = 0
        while day < len(self.combined_data):
            day_data = self.combined_data.iloc[day]
            if day_data['entry_signal']:
                long_entry = day_data['underperformer_price']
                short_entry = day_data['overperformer_price']

                long_side = 'underperformer_price'
                short_side = 'overperformer_price'

                for day_after in range(day + 1, len(self.combined_data)):
                    day_after_data = self.combined_data.iloc[day_after]
                    if day_after_data['exit_signal']:
                        long_exit = day_after_data[long_side]
                        short_exit = day_after_data[short_side]

                        long_pnl = long_exit - long_entry
                        short_pnl = short_entry - short_exit
                        gross_pnl = long_pnl + short_pnl

                        trade_data = {
                            "open_date": self.combined_data.index[day],
                            "close_date": self.combined_data.index[day_after],
                            "long_pnl": long_pnl,
                            "short_pnl": short_pnl,
                            "gross_pnl": gross_pnl
                        }
                        trades.append(trade_data)

                        day = day_after  # Move the day index to the day after the trade close
                        break

            day += 1  # Increment the day if no trade was closed

        self.trades_df = pd.DataFrame(trades)

    def calculate_statistics(self):
        win_rate = round((len(self.trades_df[self.trades_df["gross_pnl"] > 0]) / len(self.trades_df)) * 100, 2)
        t_stat, p_val = ttest_ind(self.trades_df['long_pnl'], self.trades_df['short_pnl'])
        print(f"Win Rate: {win_rate}%")
        print(f"T-Statistic: {t_stat}, P-Value: {p_val}")
        return win_rate, t_stat, p_val

    def plot_results(self):
        plt.figure(dpi=200)
        plt.xticks(rotation=45)
        plt.title(f"{self.tickers_1[0]}-{self.tickers_2[0]} Cointegration Strategy")
        plt.plot(self.trades_df["open_date"], 100 + self.trades_df["gross_pnl"].cumsum())
        plt.plot(self.trades_df["open_date"], 100 + self.trades_df["long_pnl"].cumsum())
        plt.plot(self.trades_df["open_date"], 100 + self.trades_df["short_pnl"].cumsum())
        plt.legend(["gross_pnl", "long_only_pnl", "short_only_pnl"])
        plt.ylabel("Capital")
        plt.xlabel("Date")
        plt.show()

        plt.figure(dpi=200)
        plt.xticks(rotation=45)
        plt.title(f"{self.tickers_1[0]}-{self.tickers_2[0]} Return Spread")
        plt.ylabel("% Difference in Returns")
        plt.xlabel("Date")
        plt.plot(self.combined_data.index, self.combined_data["spread"])
        plt.show()

        plt.figure(dpi=200)
        plt.xticks(rotation=45)
        plt.title(f"{self.tickers_1[0]}-{self.tickers_2[0]} Spread")
        plt.plot(self.combined_data.index, self.combined_data["spread"])
        plt.plot(self.combined_data.index, self.combined_data["upper_band"])
        plt.plot(self.combined_data.index, self.combined_data["lower_band"])
        plt.legend(["spread", "upper_bound", "lower_bound"])
        plt.show()

        plt.figure(dpi=600)
        plt.xticks(rotation=45)
        plt.title(f"{self.tickers_1[0]}-{self.tickers_2[0]} Return Performance")
        plt.plot(self.combined_data.index, self.combined_data["1_portfolio_performance"])
        plt.plot(self.combined_data.index, self.combined_data["2_portfolio_performance"])
        plt.legend([self.tickers_1[0], self.tickers_2[0]])
        plt.show()

    def run(self):
        self.preprocess_data()
        p_value, r_squared = self.check_cointegration()
        print(f"P-Value: {p_value}, R-Squared: {r_squared}")
        self.backtest()
        self.calculate_statistics()
        self.plot_results()


# Initialize and run the strategy
api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
strategy = CointegrationStrategy(tickers_1=["FDX"], tickers_2=["UPS"], api_key=api_key)
strategy.run()
