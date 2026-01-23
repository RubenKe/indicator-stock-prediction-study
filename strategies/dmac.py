import pandas as pd
import os
from pathlib import Path
import backtrader as bt

PROJECT_ROOT = Path.cwd().parent
data_dir = Path(f"{PROJECT_ROOT}/data/raw")
files = list(data_dir.glob("*"))

print(files)