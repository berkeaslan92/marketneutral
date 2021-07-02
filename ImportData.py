import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as ys
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt


class ImportPriceData:
    """
    Class to import price data from Yahoo Finance.

    Methods:
        - save_sp500_tickers:
        - get_price_data:
        - ...
    """

    def __init__(self, start_date: str, end_date: str):
        """
        Constructor for the ImportPriceData class.

        Parameters
        :param start_date: Start date to choose to import the price data.
        :param end_date: End date to choose to import the price data.

        Type
        :type start_date: str
        :type end_date: str
        """
        self.__tickers = None
        self.__price_data = None
        self.__return_data = None
        self.__start_date = pd.to_datetime(start_date)
        self.__end_date = pd.to_datetime(end_date)

    def save_sp500_tickers(self) -> list:
        """
        Method to save the SP500 tickers.

        :return: self.__tickers -> list of tickers representing the symbols in the current S&P500
        """

        # Yahoo Finance Stock info function to retrieve S&P500 Tickers
        self.__tickers = ys.tickers_sp500()

        # print(final_tickers) !works
        return self.__tickers

    def get_price_data(self, refresh_data=False):
        """
        Method to save the price data.
        If data is not available, collect it from Yahoo Finance and save it in a CSV file.
        If the data is already available, read it from the CSV file from the project directory.

        :return: self.__price_data -> Pandas Core DataFrame that contains S&P500 prices.
        """

        # Find the file name to read/write depending on refreshing
        today = date.today().strftime("%m-%d-%y")
        csv_file_name = 'price_data_' + today + '.csv'

        # If we want to refresh the data, get the new tickers and download new data with
        # the specified parameters
        if refresh_data:
            tickers = self.save_sp500_tickers()
            self.__price_data = yf.download(tickers=self.__tickers,
                                            start=self.__start_date,
                                            end=self.__end_date,
                                            interval="1d",
                                            group_by='column')['Close']

            self.__price_data.to_csv(csv_file_name, sep=',', encoding='utf-8')

            # Modify later on this line for the strptime problem reading the csv
            self.__price_data = pd.read_csv(csv_file_name)

        else:
            self.__price_data = pd.read_csv(csv_file_name)
            self.__price_data = self.__price_data.set_index('Date')

        # Remove the discontinuity in the dataset
        self.__price_data = self.remove_discontinuity_price_data()

        return self.__price_data

    def remove_discontinuity_price_data(self):
        """
        Method to remove discontinuity in prices (NaN values).
        Currently uses Pandas' fillna method (ffill) that takes the last value seen in the column.
        !TODO: use the Brownian Motion (Random Walk) or previous/next mean strategy.

        :return: self.__price_data -> Pandas Core DataFrame that contains **continuous** S&P500 prices.
        """
        # Change replacing strategy to Brownian Motion or Random Walk later on
        return self.__price_data.fillna(method='ffill').dropna(axis='columns')

    def get_returns_data(self):
        """
        Method to generate the percentage returns of the price dataset.

        :return: self.__returns_data -> Pandas Core DataFrame that contains **continuous** S&P500 returns.
        """
        if self.__price_data is None:
            self.get_price_data()

        self.__return_data = pd.DataFrame(index=self.__price_data.index.copy())
        self.__return_data = self.__price_data.pct_change()

        return self.__return_data.dropna()

    def display_trading_universe(self, returns: bool):
        """
        Method to display graph of the S&P500 (prices or returns, depending on the parameter).

        :param: metric -> str
        :return: S&P500 Graph
        """
        if returns:
            data_display = self.__return_data
        else:
            data_display = self.__price_data

        sns.lineplot(data=data_display)
        plt.show()

    def download_backtest_data(self, ticker):
        back_test_data = yf.download(tickers=ticker,
                                     start=self.__start_date,
                                     end=self.__end_date,
                                     interval="1d",
                                     group_by='column')

        return back_test_data
