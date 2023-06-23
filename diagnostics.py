
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

from PIL import Image

from funcoes_modelos import montar_dataframe_temp
from funcoes_modelos import predict_ARIMA_GARCH
from funcoes_modelos import return_exog

import warnings
warnings.filterwarnings('ignore')

from tscv import TimeBasedCV

import pickle


#########################################################################
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

#plot_acf(returns, lags = 400, zero = False)
#plt.show()
#plot_pacf(returns, lags = 400, zero = False)
#plt.show()
#plot_acf(returns.diff(1).dropna(), lags = 400, zero = False)
#plt.show()
#plot_pacf(returns.diff(1).dropna(), lags = 400, zero = False)
#plt.show()

model =  sm.tsa.statespace.SARIMAX(returns , order=(1,1,3), seasonal_order=(0,1,1,7),
                                    enforce_stationarity=False, enforce_invertibility=False, freq='D')
model = model.fit()


pred = model.forecast(1)

model.plot_diagnostics(figsize=(15, 12))
#plt.show()

print(shapiro(model.resid))

print(sm.stats.acorr_ljungbox(model.resid, return_df=True, boxpierce = True))

with open('./models/model_sarima_summary.pickle', 'wb') as file:
    f = pickle.dump(model.summary(), file)














