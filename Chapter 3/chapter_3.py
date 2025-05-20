import pandas as pd
from tqdm import tqdm
 
class Engine():
    def __init__(self, initial_cash=100_000):
        self.initial_cash = initial_cash
        self.current_idx = None
        self.data = {}
        self.strategy = None
        
    def add_data(self, data:pd.DataFrame, ticker:str):
        self.data[ticker] = data
        
    def add_strategy(self, strategy):
        self.strategy = strategy
    
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
           
    # MAJOR CHANGE IN CHAPTER 3     
    def _fill_orders(self):
        for order in self.strategy.orders:
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
                    
            if can_fill:
                t = Trade(
                    ticker = order.ticker,
                    side = order.side,
                    price= fill_price,
                    size = order.size,
                    type = order.type,
                    idx = self.current_idx)

                self.strategy.trades.append(t)
                self.strategy.cash -= t.price * t.size

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
    # ADDED IN CHAPTER 3
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
    
    # ADDED IN CHAPTER 3
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

    # ADDED IN CHAPTER 3
    def close(
        self,
        ticker
        ):
        return self.data[ticker].loc[self.current_idx]['Close']
    
    # ADDED IN CHAPTER 3
    def open(
        self,
        ticker
        ):
        return self.data[ticker].loc[self.current_idx]['Open']
    
    # ADDED IN CHAPTER 3
    def low(
        self,
        ticker):
        return self.data[ticker].loc[self.current_idx]['Low']
    
    # ADDED IN CHAPTER 3
    def high(
        self,
        ticker):
        return self.data[ticker].loc[self.current_idx]['High']
    
    # ADDED IN CHAPTER 3
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
        # ADDED IN CHAPTER 3
        limit_price=None,
        order_type='market'):

        self.idx = idx
        self.side = side
        self.size = size
        self.ticker = ticker
        self.type = 'market'
        # ADDED IN CHAPTER 3
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
        fees=0
        ):
        
        self.idx = idx
        self.fees = fees
        self.price = price
        self.side = side
        self.size = size
        self.ticker = ticker
        self.type = type
 
    def __repr__(self):
        side = "BUY" if self.size > 0 else "SELL"
        return (
            f'<Trade: {self.idx.strftime("%Y-%m-%d")} '
            f'{self.ticker} {side} '
            f'{abs(self.size)}@{self.price:.2f}>'
        )


