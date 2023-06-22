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
    st.write(plot_acf(returns, lags = 400, zero = False))
with tab2:
    st.write(plot_pacf(returns, lags = 400, zero = False))

with tab3:
    st.write(plot_acf(returns.diff(1).dropna(), lags = 400, zero = False))
with tab4:
    st.write(plot_pacf(returns.diff(1).dropna(), lags = 400, zero = False, ))

st.markdown('### Modelagem')
st.markdown('#### A seguir podemos ver a modelagem de diversos tipos de modelos, comparados atravez de uma validação cruzada de janela deslizante ')
st.markdown('#### Cada modelo foi testado 10 vezes, predizedo sempre um passo a frente a raiz do erro quadratico médio (RMSE) de cada um pode ser visto na tabela abaixo')

