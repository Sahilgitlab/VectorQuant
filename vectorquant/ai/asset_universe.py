"""
Global Asset Support & Abstraction
"""
from dataclasses import dataclass

@dataclass
class AssetUniverse:
    ticker: str
    asset_class: str # "Equity", "Crypto", "Rates", "Commodity", "FX", "Options"
    multiplier: float = 1.0
    tick_size: float = 0.01
    margin_requirement: float = 1.0 # 1.0 means fully funded required

class AssetData:
    def __init__(self, asset: AssetUniverse, prices, volumes, timestamps=None):
        self.asset = asset
        self.prices = prices
        self.volumes = volumes
        self.timestamps = timestamps or list(range(len(prices)))
        
    def get_returns(self):
        if len(self.prices) < 2: return []
        return [(self.prices[i] - self.prices[i-1]) / self.prices[i-1] for i in range(1, len(self.prices))]
        
    def get_log_returns(self):
        import math
        if len(self.prices) < 2: return []
        return [math.log(self.prices[i] / self.prices[i-1]) for i in range(1, len(self.prices))]
