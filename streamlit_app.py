#import altair as alt
#import pandas as pd
#import streamlit as st
#
#st.set_page_config(
#    page_title="Trabalho Final Séries Temporais - 2s 2023", page_icon="⬇", layout="centered"
#)
#
#
#st.title("Trabalho final Séries Temporais")
#
#st.write("Começo dos testes para criação do DashBoard")

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

from funcoes_modelos import montar_dataframe_temp
from funcoes_modelos import predict_ARIMA_GARCH
from funcoes_modelos import return_exog

import warnings
warnings.filterwarnings('ignore')


st.set_page_config('Séries Tempoais', page_icon=	':chart_with_upwards_trend:')

with st.sidebar:
    st.markdown("# ME607")
    st.markdown("Estas são as abas disponíveis para mais detalhes sobre as análises e modelagem dos dados:")

st.title(':chart_with_upwards_trend: Trabalho Final Séries Temporais' )
st.subheader('Grupo: Gabriel Ukstin Talasso - 235078 ; ....')
st.markdown("## Visão geral dos dados")
st.markdown("### Alterne entre as abas para as visualizações")

@st.cache_data
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

if data not in st.session_state:
    st.session_state['df'] = data


tab1, tab2, tab3 = st.tabs([ "Grafico da Série", 
                      "Grafico Diferenciada",
                      "Tabela dos dados"])

with tab1:
    fig = px.line(returns, title='Temperatura Média Diária', 
                  labels=({'value':'Temperatura Média', 'time':'Data'}))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

with tab2:
    fig = px.line(returns.diff(1).dropna(), title='Temperatura Média Diária - Diferenciada',
                  labels = {'value':'Diferença da temperatura', 'time':'Data'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

with tab3:
    st.write(data)

st.markdown('### Para um vislumbre da dinâmica dos dados, a seguir podemos ver os seguintes gráficos:')

tab1, tab2, tab3, tab4 = st.tabs([ "ACF - Original", 
                      "PACF - Original",
                      "ACF - Diferenciada",
                      "PACF - DIferenciada"])
with tab1:
    st.write(plot_acf(returns, lags = 400))
with tab2:
    st.write(plot_pacf(returns, lags = 400))
with tab3:
    st.write(plot_acf(returns.diff(1).dropna(), lags = 400))
with tab4:
    st.write(plot_acf(returns.diff(1).dropna(), lags = 400))

st.markdown('### Modelagem')
st.markdown('#### Preencha as entradas necessárias e aperte o botão para testar diversos modelos: ')
st.markdown('Também estão disponíveis modelos de volatilidade, implementados para avaliação de seu comportamento, apesar de não serem úteis nesse caso.')

n_cv = st.number_input('Número de splits na validação cruzada:',
                        min_value = 2, max_value=20)

st.markdown('#### Faça o seu modelo ARIMA para testar:')

ar = st.number_input('Grau da parte AR do modelo a ser testado:',
                     min_value = 1, max_value=20)
ma = st.number_input('Grau da parte MA do modelo a ser testado:',
                     min_value = 1, max_value=20)
d = st.number_input('Grau de diferenciação:',
                     min_value = 1, max_value=20)

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
           ARIMA(order = (ar,d,ma)),
           ARIMA(order = (1,1,1), seasonal_order = (1,1,1), season_length = 180)
           ]


model_names = ['Media', 'Naive', 'Drift','ExpSmo', #'HoltWin180','HoltWin30',
               'AutoARIMA','ARCH1','ARCH2', 'GARCH11', 'GARCH22', 'ARIMA-GARCH',
               'Seu ARIMA', 'sarima']

c = st.checkbox('Rodar nosso modelo conjuntamente.')
if c:
    modelos.append('ourmodel')
    model_names.append('Nosso Modelo')
    st.error('Atenção! Isso pode demorar um pouco.')


b = st.button('Rodar Modelos')

if b:
    tscv = TimeSeriesSplit(n_splits = n_cv, max_train_size= 600)
    erros = pd.DataFrame(columns = ['Model', 'm5_rmse'])

    n = 1

    for i, model in enumerate(modelos):

        model_name = model_names[i]
        rmse = []

        for train_index, test_index in tscv.split(returns):
            cv_train, cv_test = returns.iloc[train_index], returns.iloc[test_index]

            if model_name == 'ARIMA-GARCH':

                temp_train = montar_dataframe_temp(cv_train)

                predictions = predict_ARIMA_GARCH(model, temp_train, n)

            elif model_name == 'Nosso Modelo':

                temp_train = montar_dataframe_temp(cv_train)

                sarimax = sm.tsa.statespace.SARIMAX(temp_train['tavg'] , order=(1,1,1), exog = temp_train[['precip_ontem', 'precip_media_semana']],
                                        enforce_stationarity=False, enforce_invertibility=False, freq='D', simple_differencing=True).fit(low_memory=True, cov_type='none')

                predictions = sarimax.forecast(n, exog = return_exog(temp_train, n).values).values

            else:
                model = model.fit(cv_train.values)

                predictions = model.predict(n)
                predictions = predictions['mean']#[0]


            true_values = cv_test.values[0:n]
            rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))


        erros = pd.concat([erros, pd.DataFrame([{'Model': model_name,'m5_rmse': np.mean(rmse)}])],
                                  ignore_index = True)
        
    st.markdown('#### RMSE dos modelos (ordenado):')
    erros.sort_values('m5_rmse').T