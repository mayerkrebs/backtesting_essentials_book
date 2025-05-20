import pandas as pd
from tqdm import tqdm
import math
import numpy as np
import itertools
from dateutil.relativedelta import relativedelta
from datetime import timedelta

class Engine():
    def __init__(
        self, 
        initial_cash=100_000,
        warmup_periods=0
        ):
        self.initial_cash = initial_cash
        self.current_idx = None
        self.data = {}
        self.strategy = None
        self.broker_fee_model = BrokerFee()
        
        self.market_impact_model = MarketImpact(b=0)
        self.max_participation_rate = 0.1
        
        self.warmup_periods = warmup_periods

        
    def add_data(self, data:pd.DataFrame, ticker:str):
        self.data[ticker] = data
        
    def add_strategy(self, strategy):
        self.strategy = strategy
    
    def add_broker_fee_model(self, broker_fee_model):
        self.broker_fee_model = broker_fee_model

    def add_market_impact_model(self, market_impact_model):
        self.market_impact_model = market_impact_model


    def run(self):
        self.strategy.data = self.data
        self.strategy.cash = self.initial_cash
        
        self.strategy.preprocessing()

        self.strategy.data = {
            k: v.iloc[self.warmup_periods:] 
            for k, v in self.data.items()
        }
        self.data = self.strategy.data

        first_ticker = next(iter(self.data))
        for idx in tqdm(self.data[first_ticker].index):
            self.current_idx = idx
            self.strategy.current_idx = self.current_idx
            
            self._fill_orders()
            
            self.strategy.on_bar()
            
        self._close_all_positions()
            
        return self._get_stats()
           
    def _fill_orders(self):
        self.strategy.orders.sort(
            key=lambda order: order.side == "buy"
        )

        for order in self.strategy.orders:

            max_order_size = (
                self.data[order.ticker]
                .loc[self.current_idx]['Volume'] 
                * self.max_participation_rate)
            order.size = min(abs(order.size), max_order_size)

            
            if order.side == "buy":
                max_order_size = math.floor(
                    self.strategy.cash
                    / self.data[order.ticker]
                    .loc[self.current_idx]["Open"]
                )
                order.size = min(order.size, max_order_size)
            if order.side == "sell":
                max_order_size = self.strategy.position_size(order.ticker)
            order.size = min(abs(order.size), max_order_size)
            
            if order.side == 'sell':
                order.size = -order.size
            
            fill_price = self.data[order.ticker].loc[self.current_idx]['Open']
            can_fill = False
            current_open = self.data[order.ticker].loc[self.current_idx]['Open']
            if order.side == 'buy' and self.strategy.cash >= self.data[order.ticker].loc[self.current_idx]['Open'] * order.size:
                if order.type == 'limit':
                    if order.limit_price >= self.data[order.ticker].loc[self.current_idx]['Low']:
                        fill_price = min(order.limit_price, current_open)
                        can_fill = True
                elif order.type == 'market':        
                    can_fill = True 
                    
            elif order.side == 'sell' and self.strategy.position_size(order.ticker) + order.size >= 0:
                if order.type == 'limit':
                    if order.limit_price <= self.data[order.ticker].loc[self.current_idx]['High']:
                        fill_price = max(order.limit_price, current_open)
                        can_fill = True
                elif order.type == 'market':
                    can_fill = True

            current_bar = (
                self.data[order.ticker]
                .loc[self.current_idx]
            )
            
            market_impact = (
                self.market_impact_model
                .calculate(order, current_bar)
            )
            if order.side == 'buy':
                fill_price += market_impact
            else:
                fill_price -= market_impact

     
            if can_fill:
                t = Trade(
                    ticker = order.ticker,
                    side = order.side,
                    price= fill_price,
                    size = order.size,
                    type = order.type,
                    idx = self.current_idx)
                
                t.fees = self.broker_fee_model.calculate(t)
                
                t.market_impact_cost = abs(market_impact * t.size)
                
                self.strategy.trades.append(t)
                self.strategy.cash -= t.price * t.size + t.fees



        self.strategy.orders = []
        
    def _get_stats(self):
        metrics = {}
        
        final_aum = sum(
            self.data[ticker].loc[self.current_idx]['Close'] \
                        * self.strategy.position_size(ticker) 
                        for ticker in self.data) \
                        + self.strategy.cash
        total_return = 100 * (final_aum / self.initial_cash -1)
        metrics['total_return'] = total_return
        
        idx_list = self.data[next(iter(self.data))].index
        df_trades = pd.DataFrame(
            {t: 0 for t in self.strategy.tickers}, 
            index=idx_list
        )
        df_trades = df_trades.astype(float)
    
        cash_movements = (
            pd.Series(0, index=idx_list)
            .astype(float))
        cash_movements.iloc[0] = self.initial_cash
        for t in self.strategy.trades:
            df_trades.loc[t.idx, t.ticker] += t.size
            cash_movements.loc[t.idx] += (
                -t.size * t.price - t.fees
            )
            
        cash_series = cash_movements.cumsum()
        
        positions = df_trades.fillna(0).cumsum()
        prices = pd.DataFrame(
            {
            ticker: self.data[ticker]['Close'] 
            for ticker in self.data
            }
        )
        p = (positions * prices).sum(axis=1) + cash_series
        metrics['aum_series'] = p  
        
        total_years = (p.index[-1] - p.index[0]).days / 365

        metrics['annualized_returns'] = (
            (p.iloc[-1] / p.iloc[0]) ** (1 / total_years) - 1) * 100
        
        log_returns = np.log(p / p.shift(1))
        std_log_returns = log_returns.std()
        metrics['volatility'] = std_log_returns * np.sqrt(252) * 100
        
        avg_log_returns = log_returns.mean() 
        daily_sharpe_ratio = avg_log_returns / std_log_returns
        metrics['sharpe_ratio'] = daily_sharpe_ratio * np.sqrt(252)
        
        roll_max = p.cummax()
        daily_drawdown = p / roll_max - 1.0
        max_daily_drawdown = daily_drawdown.cummin()
        
        metrics['max_drawdown'] = max_daily_drawdown.min() * 100
        
        metrics['exposure'] = (1 - cash_series / p).mean() * 100
        
        metrics['trade_journal'] = self.create_trade_journal(
            self.strategy.trades
        )

        return metrics

    def _close_all_positions(self):
        for ticker in self.strategy.tickers:
            position = self.strategy.position_size(ticker)
            if position != 0:
                t = Trade(
                    ticker=ticker,
                    side="sell" if position > 0 else "buy",
                    price=self.strategy.close(ticker),
                    size=-position,
                    type="market",
                    fees=0,
                    idx=self.current_idx,
                    market_impact_cost=0
                )
                self.strategy.trades.append(t)
                self.strategy.cash -= t.price * t.size
    
    def create_trade_journal(self, trades):
        dict_trades = [t.__dict__ for t in trades]
        
        df_trades = pd.DataFrame(dict_trades).reset_index()
    
        trade_journal = []
        for ticker, orders in (
            df_trades.groupby("ticker")
            ):
            
            for idx, sell_order in orders[
                orders["side"] == "sell"
                ].iterrows():
                
                rem_qty = sell_order["size"]
                while rem_qty < 0:
                    # FIND PRIOR BUY ORDERS
                    opening_orders = orders[
                        (orders.index < idx) 
                        & (orders["size"] > 0)]
    
                    if opening_orders.empty:
                        break
    
                    # TAKE THE FIRST AVAILABLE OPENING ORDER
                    opening_order = opening_orders.iloc[0]
                    trade_size = min(
                        -rem_qty, opening_order["size"]
                        )
    
                    # ADJUST REM. QUANTITIES
                    orders.loc[
                        opening_order.name, "size"
                        ] -= trade_size
                    
                    orders.loc[
                        idx, "size"
                        ] += trade_size
                    
                    rem_qty += trade_size
    
                    # RECORD IN TRADE JOURNAL
                    trade_journal.append(
                        self._create_journal_entry(
                            opening_order,
                            sell_order,
                            trade_size,
                            "long")
                    )
        
        trade_journal = pd.DataFrame(trade_journal)
        if not trade_journal.empty:
            trade_journal["return_pct"] = (
                trade_journal["return_pct"] * 100
                )
            
            trade_journal["holding_time"] = (
                trade_journal["exit_date"]
                - trade_journal["entry_date"]
                )
            
        return trade_journal

    def _create_journal_entry(
        self,
        open_order,
        close_order,
        trade,
        type):
        
        type_multiplier = -1 if type == "short" else 1
        return {
            "type": type,
            "ticker": open_order.ticker,
            "entry_date": open_order.idx,
            "exit_date": close_order.idx,
            "entry_price": open_order.price,
            "exit_price": close_order.price,
            "size": trade,
            "return_pct": (
                type_multiplier
                * (close_order.price - open_order.price)
                / open_order.price
                ),
            "pnl": (
                type_multiplier
                * (close_order.price - open_order.price)
                * trade
                ),
            "total_fees": (
                open_order.fees
                + close_order.fees
                ),
            "market_impact_cost": (
                open_order.market_impact_cost
                + close_order.market_impact_cost
                )
        }




class Strategy():
    def __init__(
        self
        ):
        self.current_idx = None
        self.data = {}
        self.orders = []
        self.trades = []
        self.cash = None
        
        self.init()

    def init(self):
        pass

    def preprocessing(self):
        pass


    def buy(
        self,
        ticker,
        size=1
        ):
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'buy',
                size = size,
                idx = self.current_idx
            ))
 
    def sell(
        self,
        ticker,
        size=1
        ):
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'sell',
                size = -size,
                idx = self.current_idx
            ))

    def buy_limit(self,
                ticker,
                limit_price, 
                size=1):
        
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'buy',
                size = size,
                limit_price=limit_price,
                order_type='limit',
                idx = self.current_idx
            ))
    
    def sell_limit(self,
                ticker,
                limit_price,
                size=1):
        
        self.orders.append(
            Order(
                ticker = ticker,
                side = 'sell',
                size = -size,
                limit_price=limit_price,
                order_type='limit',
                idx = self.current_idx
            ))

    def rebalance(self, weights):
        total_aum = self.total_aum()
        for ticker in self.tickers:
            target_exposure = weights.get(ticker, 0)
            current_exposure = (
                self.position_size(ticker)
                * self.close(ticker)
            ) / total_aum
    
            delta = (
                target_exposure - current_exposure
                ) * total_aum
            order_size = round(
                delta / self.close(ticker), 6
                )
            if order_size > 0:
                self.buy(ticker, order_size)
            elif order_size < 0:
                self.sell(ticker, abs(order_size))


    def total_aum(self):
        aum = (
            sum(
                self.position_size(ticker) 
                * self.close(ticker)
                for ticker in self.tickers
            )
            + self.cash
        )
        return aum




    def position_size(
        self,
        ticker):
        return sum(
            [t.size for t in self.trades if t.ticker == ticker]
        )

    def close(
        self,
        ticker
        ):
        return self.data[ticker].loc[self.current_idx]['Close']
    
    def open(
        self,
        ticker
        ):
        return self.data[ticker].loc[self.current_idx]['Open']
    
    def low(
        self,
        ticker):
        return self.data[ticker].loc[self.current_idx]['Low']
    
    def high(
        self,
        ticker):
        return self.data[ticker].loc[self.current_idx]['High']
    
    def volume(
        self,
        ticker):
        return self.data[ticker].loc[self.current_idx]['Volume']

    def indicator(self, ticker, indicator):
        return self.data[ticker].loc[self.current_idx][indicator]



    @property
    def tickers(self):
        return list(self.data.keys())
    
    def on_bar(self):
        """This method will be overridden by our strategies.
        """
        pass 

    
class Order():
    def __init__(
        self,
        ticker,
        size,
        side,
        idx,
        limit_price=None,
        order_type='market'):

        self.idx = idx
        self.side = side
        self.size = size
        self.ticker = ticker
        self.type = 'market'
        self.type = order_type
        self.limit_price = limit_price

 
        
        
class Trade():
    def __init__(
        self,
        ticker,
        side,
        size,
        price,
        type,
        idx,
        fees=0,
        market_impact_cost=0
        ):
        
        self.idx = idx
        self.fees = fees
        self.price = price
        self.side = side
        self.size = size
        self.ticker = ticker
        self.type = type
        self.market_impact_cost = market_impact_cost
        
    def __repr__(self):
        side = "BUY" if self.size > 0 else "SELL"
        return (
            f'<Trade: {self.idx.strftime("%Y-%m-%d")} '
            f'{self.ticker} {side} '
            f'{abs(self.size)}@{self.price:.2f}>'
        )


class BrokerFee():
    def __init__(self, fees_pct = 0):
        self.fees_pct = fees_pct
    
    def calculate(self, trade):
        return abs(trade.size) * trade.price * self.fees_pct

class CustomBrokerFee(BrokerFee):
    def calculate(self, trade):
        fixed_fee_per_share = 0.005
        return max(1, abs(trade.size) * fixed_fee_per_share)

class MarketImpact():
    def __init__(self,
                 b=0.35,
                 sigma_year=0.30,
                 alpha=0.4):
        self.b = b
        self.sigma = sigma_year
        self.alpha = alpha
    
    def calculate(self, order, bar):
        volume = bar['Volume']
        sigma_day = self.sigma / (252 ** 0.5)
        pov = abs(order.size) / volume
        
        market_impact_pct = self.b * sigma_day *  \
            pov ** self.alpha
            
        return bar['Close'] * market_impact_pct




class ParameterOptimizer:
    def __init__(self,
                 strategy,
                 param_grid,
                 data,
                 warmup_periods=0,
                 market_impact_model=None,
                 broker_fees_model=None,
                 objective='sharpe_ratio',
                 constraints=None,
                 high_is_best=True):
        
        self.strategy = strategy
        self.param_grid = param_grid
        self.data = data
        self.warmup_periods = warmup_periods
        self.market_impact_model = market_impact_model
        self.broker_fees_model = broker_fees_model
        self.objective = objective
        self.constraints = constraints or (lambda _: True)
        self.high_is_best = high_is_best
 
    def _build_engine(self, strategy_instance):
        engine = Engine(
            warmup_periods=self.warmup_periods
            )
        if self.market_impact_model:
            engine.add_market_impact_model(
                self.market_impact_model
                )
        if self.broker_fees_model:
            engine.add_broker_fee_model(
                self.broker_fees_model
                )
 
        engine.add_strategy(strategy_instance)
        for ticker, df in self.data.items():
            engine.add_data(df, ticker)
        return engine
 
    def _evaluate_params(self, param_dict):
        if not self.constraints(param_dict):
            return None
 
        strategy_instance = self.strategy()
        strategy_instance.init(**param_dict)
        engine = self._build_engine(strategy_instance)
 
        metrics = engine.run()
        return {
            **param_dict,
            self.objective: metrics[self.objective]}
 
    def optimize(self):
        param_names = list(self.param_grid.keys())
        param_combinations = itertools.product(
            *self.param_grid.values()
            )
        results = []
 
        for combination in param_combinations:
            param_dict = dict(zip(param_names, combination))
            result = self._evaluate_params(param_dict)
            if result:
                results.append(result)
 
        best = (
            max(
            results, key=lambda x: x[self.objective]
            ) if self.high_is_best else
            min(
                results, key=lambda x: x[self.objective]
                )
            )
        return best, results



def generate_walkforward_iterations(
    start_date,
    end_date,
    in_sample_months=6,
    out_sample_months=6,
    ):
    
    iterations = []
    current_in_start = start_date
 
    while True:
        current_in_end = (
            current_in_start + 
            relativedelta(months=in_sample_months) 
            - timedelta(days=1))
        
        current_out_start = (
            current_in_end 
            + timedelta(days=1))
        
        current_out_end = (
            current_out_start 
            + relativedelta(months=out_sample_months) 
            - timedelta(days=1))
 
        if current_out_end > end_date:
            break
 
        iterations.append({
            'in_sample': [
                current_in_start,
                current_in_end],
            'out_of_sample': [
                current_out_start,
                current_out_end]
        })
        current_in_start = current_out_start
 
    return iterations




class WalkForwardOptimizer:
    def __init__(self,
                 strategy,
                 param_grid,
                 iterations,
                 data,
                 warmup_periods=0,
                 market_impact_model=None,
                 broker_fees_model=None,
                 objective='sharpe_ratio',
                 constraints=lambda x: True,
                 high_is_best=True):
 
        self.strategy = strategy
        self.param_grid = param_grid
        self.iterations = iterations
        self.data = data
        self.warmup_periods = warmup_periods
        self.market_impact_model = market_impact_model
        self.broker_fees_model = broker_fees_model
        self.objective = objective
        self.constraints = constraints
        self.high_is_best = high_is_best
        
    def run(self):
        results = []
        for iteration in self.iterations:
            in_sample_start, in_sample_end = iteration[
                'in_sample']
            out_sample_start, out_sample_end = iteration[
                'out_of_sample']
 
            train_data = {}
            test_data = {}
 
            for ticker, df in self.data.items():
                
                # Find closest valid index
                # if exact match isn't present
                in_idx = df.index.get_indexer(
                    [in_sample_start],
                    method='backfill')[0]
                out_idx = df.index.get_indexer(
                    [out_sample_start],
                    method='backfill')[0]
 
                # Shift start indices 
                # back by warmup_periods (clamped at 0)
                in_sample_start_warm = df.index[
                    max(0, in_idx - self.warmup_periods)
                    ]
                out_sample_start_warm = df.index[
                    max(0, out_idx - self.warmup_periods)
                    ]
 
                train_data[ticker] = df.loc[
                    in_sample_start_warm:in_sample_end
                    ]
                test_data[ticker] = df.loc[
                    out_sample_start_warm:out_sample_end
                    ]
 
            optimizer = ParameterOptimizer(
                strategy=self.strategy,
                param_grid=self.param_grid,
                data=train_data,
                warmup_periods=self.warmup_periods,
                objective=self.objective,
                constraints=self.constraints,
                high_is_best=self.high_is_best
            )
            best_params, all_params = optimizer.optimize()
            best_objective = best_params[self.objective]
            best_params = {
                k: v for k, v 
                in best_params.items() 
                if k in self.param_grid.keys()}
            
            engine = Engine(warmup_periods=self.warmup_periods)
            if self.market_impact_model:
                engine.add_market_impact_model(
                    self.market_impact_model
                    )
            if self.broker_fees_model:
                engine.add_broker_fee_model(
                    self.broker_fees_model
                    )
            for ticker, df in test_data.items():
                engine.add_data(df, ticker)
            
            strategy = self.strategy()
            strategy.init(**best_params)
            engine.add_strategy(strategy)
 
            for ticker, df in test_data.items():
                engine.add_data(df, ticker)
            print(best_params)
            metrics = engine.run()
            results.append({
                'in_sample_from': in_sample_start,
                'in_sample_to': in_sample_end,
                'out_of_sample_from': out_sample_start,
                'out_of_sample_to': out_sample_end,
                **best_params,
                'is_iterations': all_params,
                f'is_{self.objective}': best_objective,
                f'oos_{self.objective}':
                    metrics[self.objective],
                'aum_series': metrics['aum_series'],
                })  
        return pd.DataFrame(results)



# ADDED IN CHAPTER 8
class BrinsonAttribution:
    def __init__(self,
                 strat_weights: dict,
                 bench_weights: dict,
                 strat_returns: dict,
                 bench_returns: dict,
                 strat_classes: dict,
                 bench_classes: dict):
        
        self.strat_weights = strat_weights
        self.bench_weights = bench_weights
        self.strat_returns = strat_returns
        self.bench_returns = bench_returns
        self.strat_classes = strat_classes
        self.bench_classes = bench_classes
        self.all_classes = (
            set(strat_classes.values()) 
            | set(bench_classes.values())
        )
 
    def group_by_class(self, weights, returns, classes):
        class_weights = {c: 0.0 for c in self.all_classes}
        weighted_returns = {c: 0.0 for c in self.all_classes}
 
        for ticker, weight in weights.items():
            sector = classes.get(ticker)
            if sector is None:
                continue
            class_weights[sector] += weight
            weighted_returns[sector] +=  (
                weight 
                * returns.get(ticker, 0.0)
            )
 
        return class_weights, weighted_returns
 
    def run(self):
        strat_w, strat_wr = self.group_by_class(
            self.strat_weights, 
            self.strat_returns, 
            self.strat_classes
            )
        bench_w, bench_wr = self.group_by_class(
            self.bench_weights, 
            self.bench_returns, 
            self.bench_classes
            )
 
        allocation = 0.0
        selection = 0.0
        interaction = 0.0
 
        sector_table = {}
 
        for c in self.all_classes:
            wp = strat_w.get(c, 0.0)
            wb = bench_w.get(c, 0.0)
            rp = strat_wr.get(c, 0.0)
            rb = bench_wr.get(c, 0.0)
 
            # Avoid divide-by-zero
            rp_perc = rp / wp if wp else 0.0
            rb_perc = rb / wb if wb else 0.0
 
            allocation += (wp - wb) * rb_perc
            selection += wb * (rp_perc - rb_perc)
            interaction += (wp - wb) * (rp_perc - rb_perc)
 
            sector_table[c] = {
                'Portfolio Weight': wp,
                'Benchmark Weight': wb,
                'Portfolio Return': rp_perc,
                'Benchmark Return': rb_perc
            }
 
        excess = (
            sum(strat_wr.values()) 
            - sum(bench_wr.values())
            )
 
        return {
            'Attribution': {
                'Allocation': allocation,
                'Selection': selection,
                'Interaction': interaction,
                'Total Excess': excess
            },
            'Sector Table': sector_table
        }
 
