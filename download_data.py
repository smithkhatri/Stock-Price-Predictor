import yfinance as yf

data = yf.download("AAPL", period="5y", interval="1d")

data.to_csv("AAPL.csv")

print(data.head())