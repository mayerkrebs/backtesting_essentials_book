import pandas as pd
from tqdm import tqdm
import math
class Engine():
    def __init__(self, initial_cash=100_000):
        self.initial_cash = initial_cash
        self.current_idx = None
        self.data = {}
        self.strategy = None
        # ADDED IN CHAPTER 4
        self.broker_fee_model = BrokerFee()
        
        # ADDED IN CHAPTER 4        
        self.market_impact_model = MarketImpact(b=0)
        self.max_participation_rate = 0.1

        
    def add_data(self, data:pd.DataFrame, ticker:str):
        self.data[ticker] = data
        
    def add_strategy(self, strategy):
        self.strategy = strategy
    
    # ADDED IN CHAPTER 4
    def add_broker_fee_model(self, broker_fee_model):
        self.broker_fee_model = broker_fee_model

    # ADDED IN CHAPTER 4
    def add_market_impact_model(self, market_impact_model):
        self.market_impact_model = market_impact_model


    def run(self):
        self.strategy.data = self.data
        self.strategy.cash = self.initial_cash

        first_ticker = next(iter(self.data))
        for idx in tqdm(self.data[first_ticker].index):
            self.current_idx = idx
            self.strategy.current_idx = self.current_idx
            
            self._fill_orders()
            
            self.strategy.on_bar()
        return self._get_stats()
           
    def _fill_orders(self):
        for order in self.strategy.orders:

            # ADDED IN CHAPTER 4
            max_order_size = (
                self.data[order.ticker]
                .loc[self.current_idx]['Volume'] 
                * self.max_participation_rate)
            order.size = min(abs(order.size), max_order_size)

            # END OF ADDITION
            
            # ADDED IN CHAPTER 4
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
            # END OF ADDITION
            
            # ADDED IN CHAPTER 4
            if order.side == 'sell':
                order.size = -order.size
            # END OF ADDITION
            
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

            # ADDED IN CHAPTER 4
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
            # END OF ADDITION

     
            if can_fill:
                t = Trade(
                    ticker = order.ticker,
                    side = order.side,
                    price= fill_price,
                    size = order.size,
                    type = order.type,
                    idx = self.current_idx)
                
                # ADDED IN CHAPTER 4
                t.fees = self.broker_fee_model.calculate(t)
                
                # ADDED IN CHAPTER 4
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
        return metrics

    
class Strategy():
    def __init__(
        self
        ):
        self.current_idx = None
        self.data = {}
        self.orders = []
        self.trades = []
        self.cash = None
    
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
        # ADDED IN CHAPTER 4
        market_impact_cost=0
        ):
        
        self.idx = idx
        self.fees = fees
        self.price = price
        self.side = side
        self.size = size
        self.ticker = ticker
        self.type = type
        # ADDED IN CHAPTER 4
        self.market_impact_cost = market_impact_cost
        
    def __repr__(self):
        side = "BUY" if self.size > 0 else "SELL"
        return (
            f'<Trade: {self.idx.strftime("%Y-%m-%d")} '
            f'{self.ticker} {side} '
            f'{abs(self.size)}@{self.price:.2f}>'
        )


# ADDED IN CHAPTER 4
class BrokerFee():
    def __init__(self, fees_pct = 0):
        self.fees_pct = fees_pct
    
    def calculate(self, trade):
        return abs(trade.size) * trade.price * self.fees_pct

# ADDED IN CHAPTER 4
class CustomBrokerFee(BrokerFee):
    def calculate(self, trade):
        fixed_fee_per_share = 0.005
        return max(1, abs(trade.size) * fixed_fee_per_share)

# ADDED IN CHAPTER 4
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
