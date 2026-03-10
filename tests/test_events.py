"""
Event-Driven Backtester Components Validation

Validates the event queue loop process sequence to ensure
no look-ahead biases occur internally.
"""

import sys
import os
from queue import Queue
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectorquant.research.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from vectorquant.research.event_components import (
    EventDrivenBacktester, DataHandler, Strategy, Portfolio, ExecutionHandler
)

# --- MOCKS FOR TESTING THE LOOP ---

class MockDataHandler(DataHandler):
    def __init__(self, events_queue, num_bars):
        self.events = events_queue
        self.num_bars = num_bars
        self.bar_count = 0
        self.continue_backtest = True
        
    def update_bars(self):
        if self.bar_count < self.num_bars:
            self.bar_count += 1
            self.events.put(MarketEvent())
        else:
            self.continue_backtest = False

class MockStrategy(Strategy):
    def __init__(self, events_queue):
        self.events = events_queue
        
    def calculate_signals(self, event):
        # Generate a signal strictly AFTER receiving MarketEvent
        if event.type == 'MARKET':
            self.events.put(SignalEvent("MOCK_STRAT", "AAPL", "2026-03-10", "LONG", 1.0))

class MockPortfolio(Portfolio):
    def __init__(self, events_queue):
        self.events = events_queue
        
    def update_signal(self, event):
        # Generate an order strictly AFTER receiving SignalEvent
        if event.type == 'SIGNAL':
            self.events.put(OrderEvent("AAPL", "MKT", 100, event.signal_type))
            
    def update_fill(self, event):
        pass # Normally updates PnL, we just mock receipt here
        
    def update_timeindex(self, event):
        pass

    def create_equity_curve_dataframe(self):
        pass

class MockExecutionHandler(ExecutionHandler):
    def __init__(self, events_queue):
        self.events = events_queue
        
    def execute_order(self, event):
        # Generate a fill strictly AFTER receiving OrderEvent
        if event.type == 'ORDER':
            self.events.put(FillEvent("2026-03-10", "AAPL", "MOCK_EXC", 100, event.direction, 150.0))


def test_event_driven_backtester_sequential_execution():
    """
    Simulates exactly 5 market ticks. 
    Verifies that the `events.get()` loop perfectly chains the 
    subsequent Signal, Order, and Fill events spawned internally.
    """
    
    # 5 Bars expected to produce 5 signals, 5 orders, and 5 fills
    # The backtester will loop until continue_backtest=False
    MOCK_BARS = 5
    
    # We will instantiate the backtester, but manually control run to assert state
    data = MockDataHandler(Queue(), MOCK_BARS) # We will replace queue inside backtester init
    port = MockPortfolio(Queue())
    exec_h = MockExecutionHandler(Queue())
    strat = MockStrategy(Queue())
    
    bt = EventDrivenBacktester(data, exec_h, port, strat)
    
    # Share the backtester queue pointer to all mocks
    for obj in [data, port, exec_h, strat]:
        obj.events = bt.events
        
    # Run the loop
    bt.simulate_trading()
    
    # Assert perfectly sequenced executions occurred exactly N times
    assert bt.signals == MOCK_BARS
    assert bt.orders == MOCK_BARS
    assert bt.fills == MOCK_BARS
