"""
Event Definitions for Event-Driven Backtesting

This module dictates the communication protocol between the various
components of the research infrastructure. The components communicate
purely by passing `Event` objects across an event queue.
"""

class Event:
    """
    Base class providing an interface for all subsequent 
    (inherited) events.
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update.
    This triggers the Strategy object to generate signals.
    """
    def __init__(self):
        self.type = 'MARKET'


class SignalEvent(Event):
    """
    Handles the event of a strategy making a decision to generate
    an order (a signal). The Portfolio object listens for this 
    event to size the position and place an order.
    """
    def __init__(self, strategy_id, symbol, datetime, signal_type, strength):
        """
        Initialises the SignalEvent.

        Parameters:
        strategy_id - The unique identifier for the strategy generating the signal.
        symbol      - The ticker symbol, e.g. 'AAPL'.
        datetime    - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT'.
        strength    - A parameter representing the confidence in the signal.
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The execution system will process the order based on capacity 
    and output a FillEvent.
    """
    def __init__(self, symbol, order_type, quantity, direction):
        """
        Initialises the OrderEvent.

        Parameters:
        symbol     - The instrument to trade.
        order_type - 'MKT' (Market) or 'LMT' (Limit, requires price processing).
        quantity   - Non-negative integer for quantity.
        direction  - 'BUY' or 'SELL'.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        print(f"Order: Symbol={self.symbol}, Type={self.order_type}, "
              f"Quantity={self.quantity}, Direction={self.direction}")


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage/execution handler. Stores the quantity, 
    price, and commission of the fill.
    """
    def __init__(self, timeindex, symbol, exchange, quantity, 
                 direction, fill_cost, commission=None):
        """
        Initialises the FillEvent.

        Parameters:
        timeindex  - The bar-resolution timestamp when the order was filled.
        symbol     - The instrument which was filled.
        exchange   - The exchange where the order was filled.
        quantity   - The filled quantity.
        direction  - The direction of fill ('BUY' or 'SELL')
        fill_cost  - The execution price.
        commission - Optional commission calculation.
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost

        # Calculate commission if not provided
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        """
        Simulates standard Interactive Brokers fees.
        Max(1.00, 0.005 * quantity) for USD equities.
        """
        full_cost = 0.005 * self.quantity
        return max(1.0, full_cost)
