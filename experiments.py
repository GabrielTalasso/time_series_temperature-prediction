
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import matplotlib.pyplot as plt
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

from funcoes_modelos import montar_dataframe_temp
from funcoes_modelos import predict_ARIMA_GARCH
from funcoes_modelos import return_exog

import warnings
warnings.filterwarnings('ignore')

from tscv import TimeBasedCV


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


modelos = [HistoricAverage(),
           Naive(),
          # SeasonalNaive(365),
          # SeasonalNaive(30),
           RandomWalkWithDrift(),
           SimpleExponentialSmoothing(0.9),
           #HoltWinters(season_length=180, error_type='A'),
           #HoltWinters(season_length=30, error_type='A') ,
           AutoARIMA(),
           ARCH(p = 1),
           ARCH(p = 2),
           GARCH(1,1),
           GARCH(2,2),
           [AutoARIMA(), GARCH(2, 2)],
           #SARIMAX(returns.values, order=(1,1,1), seasonal_order=(1,1,1, 365)),
           ARIMA(order = (1,1, 1)),
           ARIMA(order = (1,1,3), seasonal_order = (0,1,1), season_length = 7)
           ]


model_names = ['Media', 'Naive', 'Drift','ExpSmo', #'HoltWin180','HoltWin30',
               'AutoARIMA','ARCH1','ARCH2', 'GARCH11', 'GARCH22', 'ARIMA-GARCH',
               'Seu ARIMA', 'sarima']

modelos.append('ourmodel')
model_names.append('ARIMAX-precp')



datatrain = data.reset_index()[['time', 'tavg']]
datatrain['time'] = pd.to_datetime(datatrain['time'])

from tscv import TimeBasedCV
n_cv = 5

#tscv = TimeSeriesSplit(n_splits = n_cv, max_train_size= 740, )
tscv = TimeBasedCV(train_period= len(data) - n_cv,
                test_period=1,
                freq='days')
erros = pd.DataFrame(columns = ['Model', 'm5_rmse'])
n = 1

for i, model in enumerate(modelos):

    model_name = model_names[i]
    rmse = []
    for train_index, test_index in tscv.split(data = datatrain , date_column='time'):
        cv_train, cv_test = returns.iloc[train_index], returns.iloc[test_index]

        if model_name == 'ARIMA-GARCH':

            temp_train = montar_dataframe_temp(cv_train)

            predictions = predict_ARIMA_GARCH(model, temp_train, n)

        elif model_name == 'ARIMAX-precp':

            temp_train = montar_dataframe_temp(cv_train)

            sarimax = sm.tsa.statespace.SARIMAX(temp_train['tavg'] , order=(3,0,1), exog = temp_train[['precip_ontem', 'precip_media_semana']],
                                    enforce_stationarity=False, enforce_invertibility=False, freq='D', simple_differencing=True).fit(low_memory=True, cov_type='none')
            
            #mod = sm.tsa.arima.ARIMA(temp_train['tavg'], order=(3, 0, 1), seasonal_order=(0,1,0,365))
            #res = mod.fit(method='innovations_mle', low_memory=True, cov_type='none')

            predictions = sarimax.forecast(n, exog = return_exog(temp_train, n).values).values
            #predictions = res.forecast(n).values
        else:
            model = model.fit(cv_train.values)

            predictions = model.predict(n)
            predictions = predictions['mean']#[0]


        true_values = cv_test.values[0:n]
        rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))


    erros = pd.concat([erros, pd.DataFrame([{'Model': model_name,'m5_rmse': np.mean(rmse)}])],
                                ignore_index = True)


print(erros.sort_values('m5_rmse').T)
erros.sort_values('m5_rmse').T.to_csv(f'comparacao_cv_{n_cv}.csv')