from ImportData import ImportPriceData
from PairValidator import PairValidate
from PairPotential import PairClusters
from PairTrade import StatArb
import backtrader as bt
from statisarb import OpticsBacktest
from KalmanFilterArb import KalmanFilterTrade
from kalmanfilter import KalmanFiltrage

import pandas as pd

from backtesting import Backtest
from backtesting.test import GOOG

if __name__ == '__main__':
    start_date = "2020-05-01"
    end_date = "2021-05-11"
    # Import the prices

    import_price = ImportPriceData(start_date=start_date, end_date=end_date)

    prices_sp500 = import_price.get_price_data(refresh_data=False)
    returns_sp500 = import_price.get_returns_data()

    # Form the clusters
    selector: PairClusters = PairClusters(prices_sp500, returns_sp500)
    selector.optics()
    potential_pairs = selector.create_potential_pairs(display_pairs_info=False)

    # Create the strategy validator
    validator = PairValidate(price_data=prices_sp500, pairs_data=potential_pairs)
    validator.apply_filters()
    validated_pairs_data = validator.get_validated_pairs_data()
    validated_spread_data = validator.get_validated_spread_data()
    validated_pairs = validator.get_validated_pairs_symbols()

    # Start building the strategy
    strategy: StatArb = StatArb(validated_pairs_price_data=validated_pairs_data,
                                validated_pairs_spread_data=validated_spread_data)
    strategy.get_expected_spread_pct_change()
    strategy.obtain_potential_thresholds()
    # strategy.visualize_market_positions()

    pair = validated_pairs[2]
    pair_symbol = strategy.get_pair_symbols()[2]

    # Prepare pair 1
    price1 = prices_sp500[pair[0]].to_frame('Close')
    price1.index = pd.to_datetime(price1.index)

    # Prepare pair 2
    price2 = prices_sp500[pair[1]].to_frame('Close')
    price2.index = pd.to_datetime(price2.index)


    """
                                                 ############## PACKAGE MEASUREMENTS ###############
                                                 
                                                 
    
    # Set the Kalman Filtrage object to get the measurement errors and measurement error variances // skip first 10 days for correct measurements
    n_skip = 10
    klmf = KalmanFiltrage(x=price1,
                          y=price2)
    dynamic_beta = klmf.get_dynamic_beta()
    static_beta = klmf.get_static_beta()
    e_t, sqrt_Qt = klmf.get_measurement_error_metrics(dynamic_betas=dynamic_beta)
    klmf_trade: KalmanFilterTrade = KalmanFilterTrade(price1=price1,
                                                      price2=price2,
                                                      e_t=e_t,
                                                      sqrt_Qt=sqrt_Qt)
    klmf_trade.visualize_measurement_predictions()
    
    
    klmf: KalmanFilterTrade = KalmanFilterTrade(price1=price1,
                                                price2=price2)
                                                
                                                klmf_trader: KalmanFilterTrade = KalmanFilterTrade(price1=price1,
                                                       price2=price2,
                                                       e_t=e_t, Q_t=Q_t,
                                                       predictions=measurement_predictions)
    static_beta, dynamic_betas, state_covs = klmf_trader.kalman_filter_betas()
    klmf_trader.visualize_measurement_predictions(dynamic_betas=dynamic_betas)
    # Visualize kalman betas
    
    klmf.visualize_kalman_betas(static_beta=static_beta,
                                dynamic_beta=dynamic_betas,
                                pair_symbol=pair_symbol)
    
    # State prediction measurement errors
    klmf.visualize_measurement_predictions(dynamic_beta=dynamic_betas,
                                           state_covs=state_covs)
    """

    """
    Backtester class here.
    
    backtester: OpticsBacktest = OpticsBacktest(stock1_price_data=stock1,
                                                stock2_price_data=stock2,
                                                alpha_long=strategy.get_alpha_long(pair_symbol),
                                                alpha_short=strategy.get_alpha_short(pair_symbol))
    backtester.visualize_market_positions()
    """