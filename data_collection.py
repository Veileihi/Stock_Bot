import pandas as pd
import numpy as np
import neural_stock_bot as sb

# read data from csv into pandas dataframe
goog = pd.read_csv('NN-Stock-Bot/GOOG.csv')
# drop uneccesry columns
goog = goog.drop(['Date', 'Adj Close'], axis=1)
goog.Volume = goog.Volume / 1000000.0
# convert to numpy array
goog = np.array(goog)
goog = goog / 1000
goog = goog[-400 :]


goog_test = sb.StockBot(8, 5, 5, 0.01)
goog_test.train(goog)

predictions = goog_test.predict(goog, 100, 10)
actual = goog[-10 :]
errors = actual - predictions
print(errors) 



