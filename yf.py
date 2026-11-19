import yfinance as yf
import pandas as pd

STOCK_download = yf.download(tickers="8316.T", period="4d", interval="5m")
print(STOCK_download)
pd.DataFrame(STOCK_download).to_csv("8316.T.csv")
