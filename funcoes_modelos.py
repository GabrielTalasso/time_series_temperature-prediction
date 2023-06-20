import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import json
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from statsforecast.models import HistoricAverage
from statsforecast.models import Naive
from statsforecast.models import RandomWalkWithDrift
from statsforecast.models import SeasonalNaive
from statsforecast.models import SimpleExponentialSmoothing
from statsforecast.models import HoltWinters
from statsforecast.models import AutoARIMA
from statsforecast.models import ARIMA
from statsforecast.models import GARCH
from statsforecast.models import ARCH

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from scipy.stats import shapiro
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from itertools import product


import warnings
warnings.filterwarnings('ignore')

def read_data():
        # Set time period
    start = datetime(2010, 1, 1)
    end = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    # Create Point for Vancouver, BC
    vancouver = Point(49.2497, -123.1193, 70)
    #campinas = Point(-22.9056, -47.0608, 686)
    #saopaulo = Point(-23.5475, -46.6361, 769)

    # Get daily data for 2018
    data = Daily(vancouver, start, end)
    data = data.fetch()
    data = data[['tavg', 'prcp']]

    return data

data = read_data()
returns = data['tavg']

def montar_dataframe_temp(returns):

  temp = pd.DataFrame(returns)
  temp['precip_ontem'] = data['prcp'].shift(1)
  temp['precip_media_semana'] = temp['precip_ontem'].rolling(7).mean()
  temp = temp.dropna(axis = 0)
  return temp

def predict_ARIMA_GARCH(models, temp_train, n):
  model = models[0]
  model2 = models[1]

  sarimax = sm.tsa.statespace.SARIMAX(temp_train['tavg'] , order=(1,1,1),
                                enforce_stationarity=False, enforce_invertibility=False, freq='D').fit()

  resid = sarimax.resid.values

  garch = model2.fit(resid)

  pred1 = sarimax.forecast(n, exog = return_exog(temp_train, n).values).values
  pred2 = garch.predict(n)

  predictions = pred1 - pred2['mean']

  return predictions
def return_exog(temp, n):

  exog = pd.DataFrame(columns = ['precip_ontem', 'precip_media_semana'])

  exog['precip_ontem'] = np.ones(n)*temp.iloc[-1]['precip_ontem']

  exog['precip_media_semana'] = np.ones(n)*temp.iloc[-1]['precip_media_semana']

  return exog