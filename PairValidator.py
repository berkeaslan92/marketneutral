import pandas as pd
from statsmodels.tsa.stattools import coint
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import numpy as np
import os
from datetime import datetime
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
sns.set_style('darkgrid')


class PairValidate:
    """
    Source Paper: http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf [Statistical Arbitrage using ML]
    Pair Trading overview:
    Pair Trading is a popular market-neutral strategy used by hedge-funds, it focuses on the relative prices
    of the securities. The general idea is to long (buy) the undervalued paper and sell (short) the overvalued one.
    The profit exists if the pair's price convergence to a certain mean value.
    The technique that is going to be used in this project is an Unsupervised Learning algorithm called OPTICS.

    1. Select a pair of assets that their prices were historically proven to move together.
    2. Assuming that the relationship stays the same in the future, the spread between two prices is monitored.
    3. Long/Short positions are taken like explained in the strategy overview.
    4. Positions are closed when convergence occurs.

    The most common approaches to select the proper pairs are: (1) Distance, (2) Correlation, (3) Cointegration.

    Trading Model:
    1. Calculate pair's spread, mean and standard-deviation.
    2. Define the threshold to take the long/short and exit position.
    3. Monitor the evolution of the spread and count the threshold triggers.

    Pair selection criteria:
    1. Engle-Granger test to see if the two securities forming the pair are cointegrated.
    2. Hurst Exponent value -> 0 < H < 0.5 to test for the mean-reversion.
    3. 1 < Half-life < 252 (1y of trading day)
    4. Spread-mean cross > 12
    """

    def __init__(self, price_data: pd.DataFrame, pairs_data):
        self.__price_data = price_data
        self.__pairs_data = np.array(pairs_data)
        self.__pair_diff = None
        self.__validated_pairs = None
        self.__validated_pairs_diff = None

    @staticmethod
    def engel_filter(self,
                     stock1: str,
                     stock2: str,
                     critical_pvalue: float = 0.01) -> bool:
        """
        Method to test for cointegration of the pairs formed by the clusters.

        Test function source: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html
        The Null hypothesis is that there is no cointegration, the alternative hypothesis is that there is
        cointegrating relationship. If the pvalue is small, below a critical size, then we can reject the hypothesis
        that there is no cointegrating relationship.

        :return: list -> List of the pairs that are cointegrated.
        """

        if coint(self.__price_data[stock1], self.__price_data[stock2])[1] < critical_pvalue:
            return True
        elif coint(self.__price_data[stock2], self.__price_data[stock1])[1] < critical_pvalue:
            return True
        else:
            return False

    @staticmethod
    def hurst_filter(self,
                     stock1: str,
                     stock2: str,
                     hurst_criteria: float = 0.5) -> bool:
        """
        Method that returns the Hurst Exponent of the Prices
        Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/
        :param: price_data -> Price dataset of the Trading Universe.
        :return: Hurst Exponent.
        """
        pair_symbol = stock1 + '-' + stock2
        pair_diff = list(self.__pair_diff[pair_symbol])

        # Range of lag values
        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        tau = [sqrt(std(subtract(pair_diff[lag:], pair_diff[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = polyfit(log(lags), log(tau), 1)

        # Hurst coefficient
        hurst = poly[0] * 2.0

        # Return criteria for the Hurst Exponent
        if hurst <= hurst_criteria:
            return True
        else:
            return False

    @staticmethod
    def half_life_filter(self,
                         stock1: str,
                         stock2: str,
                         min_half_life: int = 1,
                         max_half_life: int = 252) -> bool:
        """
        Method to test whether the half-life of the series is optimal.
        Source: https://quant.stackexchange.com/questions/25086/calculating-half-life-of-mean-reverting-series-with-python

        :param self:
        :param max_half_life: Max 252 days to revert (1 year or trading day)
        :param min_half_life: Min 1 day to revert
        :param stock1: Stock 1 of the pair
        :param stock2: Stock 2 of the pair
        :return: Bool -> Return whether the half-life is between 1-252 Trading days
        """
        pair_symbol = stock1 + '-' + stock2
        spread_data = list(self.__pair_diff[pair_symbol])

        spread_lag = np.roll(spread_data, 1)
        spread_lag[0] = 0
        spread_ret = spread_data - spread_lag
        spread_ret[0] = 0

        # Add intercept terms variable for regression
        spread_lag2 = sm.add_constant(spread_lag)

        model = sm.OLS(spread_ret, spread_lag2)
        res = model.fit()

        # Calculate the half-life
        halflife = -log(2) / res.params[1]

        # Return the half-life
        return min_half_life < halflife < max_half_life

    @staticmethod
    def mean_cross_filter(self,
                          stock1: str,
                          stock2: str,
                          cross_criteria_year: int = 12,
                          trading_days_year: int = 252) -> bool:
        """
        Method that verifies whether the stock spread crossed its mean at least 12 times over a year.
        :param: stock1: First stock in pair.
        :param: stock2: Second stock in pair.
        :return: Bool -> Whether the stock spread crossed its mea at least 12 times a year.
        """
        # Create symbol string, get spread data and its mean
        pair_symbol = stock1 + '-' + stock2
        spread_data = self.__pair_diff[pair_symbol]
        spread_mean = spread_data.mean()

        # Get the differences between dates
        start_date = datetime.strptime(spread_data.index[0], '%Y-%m-%d')
        end_date = datetime.strptime(spread_data.index[-1], '%Y-%m-%d')
        time_delta_days = (end_date - start_date).days

        # Create criteria to pass
        cross_criteria = time_delta_days * cross_criteria_year / trading_days_year

        # Condition the loop
        cross_counter = 0
        for i in range(len(spread_data.values) - 1):
            if spread_data.values[i] < spread_mean < spread_data.values[i + 1] or \
                    spread_data.values[i] > spread_mean > spread_data.values[i + 1]:
                cross_counter = cross_counter + 1

        return cross_counter > cross_criteria

    def create_pair_differences(self):
        """
        Method to create a dataframe of the price difference for potential pairs.
        :return: self.__pair_diff: pd.DataFrame -> DataFrame of the price differences.
        """

        # Create an empty dataframe of pair differences, we will append this later.
        pair_string_names = []
        pair_price_diff = []

        for pair in self.__pairs_data:
            # Choose both stocks from each pair
            stock_symbol_1 = pair[0]
            stock_symbol_2 = pair[1]

            # Create a string that symbolizes the pair and add it to a list of strings
            pair_string = str(stock_symbol_1) + '-' + str(stock_symbol_2)
            pair_string_names.append(pair_string)

            # Get both stock prices from the price dataset
            stock_price1 = self.__price_data[stock_symbol_1]
            stock_price2 = self.__price_data[stock_symbol_2]
            pair_diff = stock_price2 - stock_price1
            pair_price_diff.append(pair_diff)

        # Concat all the pairs into the pair differences attribute in class and set column names
        self.__pair_diff = pd.concat([pd.Series(pair_prices) for pair_prices in pair_price_diff], axis=1)
        self.__pair_diff.columns = pair_string_names

        return self.__pair_diff

    def apply_filters(self):
        """
        Method to apply all the filters created in this class.
        -> Engel-Granger test for the *cointegration*
        -> Hurst Exponent test for the *mean-reversion*
        -> Half-life test for the *time* that the series will take to mean-revert.
        :return: hurst: pd.DataFrame -> Indicates the Hurst Exponent values of stocks
        """
        hurst_cut = 0
        coint_cut = 0
        half_life_cut = 0
        mean_cross_cut = 0

        # Create an empty list for pairs that pass the filter tests
        validated_pairs = []

        # Create all the pairs combination
        self.create_pair_differences()

        # Print the number of potential pairs
        print(f"Number of potential pairs in before filter: {len(self.__pairs_data)}")

        for pair in self.__pairs_data:
            # Select the stocks from the pair
            stock1 = pair[0]
            stock2 = pair[1]

            # Test the hurst filter
            if self.hurst_filter(self, stock1=stock1, stock2=stock2):
                hurst_cut += 1
                if self.engel_filter(self, stock1=stock1, stock2=stock2):
                    coint_cut += 1
                    if self.half_life_filter(self, stock1=stock1, stock2=stock2):
                        half_life_cut += 1
                        if self.mean_cross_filter(self, stock1=stock1, stock2=stock2):
                            mean_cross_cut += 1
                            validated_pairs.append([stock1, stock2])

        print(f"Hurst filter pass: {hurst_cut}")
        print(f"Co-integration filter pass: {coint_cut}")
        print(f"Half-life filter pass: {half_life_cut}")
        print(f"Mean-cross filter pass: {mean_cross_cut}")
        print(f"Final Number of validated pairs: {len(validated_pairs)}")
        print("The final validated pairs are: ")
        print(validated_pairs)

        # Save it to the attribute
        self.__validated_pairs = validated_pairs
        self.__validated_pairs_diff = self.__pair_diff[self.symbolize_pairs(self.__validated_pairs)]

    def visualize_pairs(self, individualize: bool = False, fig_size: tuple = (20, 12), normalize: bool = True):
        """
        Method to visualize the spread's movement start/end date.
        """

        if not individualize:
            # Create the plot dataframe
            if normalize:
                df = pd.DataFrame(StandardScaler().fit_transform(self.__validated_pairs_diff))
                df.index = self.__validated_pairs_diff.index
                df.columns = self.__validated_pairs_diff.columns
            else:
                df = self.__validated_pairs_diff
            plot_df = pd.melt(df, ignore_index=False)
            plot_df.columns = ['Pairs', 'Price differences']

            # Create the figure with size and axes, modify x-axis frequency
            fig, ax = plt.subplots(figsize=fig_size)
            fmt_half_year = mdates.MonthLocator(interval=3)
            ax.xaxis.set_major_locator(fmt_half_year)

            # Plot the data
            sns.lineplot(data=plot_df,
                         x="Date",
                         y="Price differences",
                         hue='Pairs')

            # Add title, x-y-labels
            plt.title('Validated pairs delta')
            plt.xlabel('Dates')
            plt.ylabel('Pair Delta')

            plt.show()
        else:
            # Create the proper axis
            number_of_validated_pairs = len(self.__validated_pairs)
            fig = plt.subplots(number_of_validated_pairs, 1, sharex=True, figsize=fig_size)
            plt.suptitle('Individual pairs delta')
            ax_counter = 1

            # Find the pairs
            for pair in self.__validated_pairs:
                stock1 = pair[0]
                stock2 = pair[1]

                plot_df = pd.melt(self.__price_data[pair], ignore_index=False)
                plot_df.columns = ['Pairs', 'Price differences']
                plt.subplot(number_of_validated_pairs, 1, ax_counter)
                ax = sns.lineplot(data=plot_df, x=plot_df.index, y='Price differences', hue='Pairs')
                fmt_half_year = mdates.MonthLocator(interval=3)
                ax.xaxis.set_major_locator(fmt_half_year)

                ax_counter += 1

            plt.show()

    @staticmethod
    def symbolize_pairs(list_of_pair_string: str) -> list:
        """
        Method to convert the pairs list into "Pair1-Pair2" string symbol for dataframes.
        :param list_of_pair_string: List of list of two strings for the pairs.
        :return: List of strings with the specified format.
        """
        symbolized_pairs = []
        for pair in list_of_pair_string:
            symbolized_pairs.append(pair[0] + '-' + pair[1])

        return symbolized_pairs

    def get_validated_pairs_data(self):
        """
        Method to return the price dataframe of validated pairs.
        :return: pd.DataFrame -> DataFrame of validated price pairs.
        """
        validated_pairs_prices = []
        for pair in self.__validated_pairs:
            stock1 = pair[0]
            stock2 = pair[1]
            validated_pairs_prices.append(self.__price_data[stock1])
            validated_pairs_prices.append(self.__price_data[stock2])

        validated_pairs_prices = pd.DataFrame(validated_pairs_prices).T

        return validated_pairs_prices

    def get_validated_spread_data(self):
        """
        Method to return the spread dataframe of validated pairs.
        :return: pd.DataFrame -> DataFrame of validated spread values.
        """
        validated_pairs_spread = []
        symbols = self.symbolize_pairs(self.__validated_pairs)
        for symbol in symbols:
            validated_pairs_spread.append(self.__validated_pairs_diff[symbol])

        validated_pairs_spread = pd.DataFrame(validated_pairs_spread).T

        return validated_pairs_spread

    def get_validated_pairs_symbols(self):
        """
        Method to get the validated pairs symbols.
        :return: List of Strings. Returns the validated pairs symbols.
        """
        return self.__validated_pairs