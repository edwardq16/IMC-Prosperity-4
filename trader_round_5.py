import numpy as np
import math
from datamodel import TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        traderData = ""
        conversions = 0
        return result, conversions, traderData