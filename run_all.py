import backtrader as bt
import datetime
import pandas as pd
from strategies import ALL_STRATEGIES
import yaml

# open the config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
