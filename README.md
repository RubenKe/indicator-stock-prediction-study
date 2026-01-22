# Algorithmic Trading & AI Research Framework

## Project description

This repository contains the codebase developed for the research paper:

**“In welke mate kunnen algoritmen en artificiële intelligentie bijdragen aan het voorspellen van aandelenprijzen?”**

The goal of this project is to systematically evaluate whether:
- traditional algorithmic trading strategies
- and artificial intelligence models  

can contribute to predicting stock price movements and achieving effective or profitable trading strategies.

The project is designed as a **reproducible research framework**, not as a live trading system.

---

## Research goals

- Evaluate classic technical trading strategies (non-AI)
- Compare their performance with AI-based models
- Test strategies across:
  - multiple markets (stocks, crypto, forex, indices)
  - multiple time intervals (e.g. 15m, 1h, 1d)
  - multiple parameter combinations
- Store all results in a structured format for later analysis

---

## Project structure

project/
│
├─ strategies/ # Trading strategies (logic only)
│ ├─ dmac.py
│ ├─ rsi.py
│ ├─ macd.py
│ └─ init.py
│
├─ experiments/ # Execution and experiments
│ ├─ config.yaml
│ ├─ run_all.py
│
├─ utils/ # Shared utilities
│ ├─ data_loader.py
│ ├─ backtester.py
│ ├─ results_logger.py
│
├─ data/
│ ├─ raw/ # Downloaded raw market data
│ └─ processed/ # Cleaned and processed data
│
├─ database/
│ └─ results.parquet # Aggregated experiment results
│
├─ .gitignore
├─ README.md
└─ requirements.txt