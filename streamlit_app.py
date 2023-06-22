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

from PIL import Image

import warnings
warnings.filterwarnings('ignore')


st.set_page_config('Séries Tempoais', page_icon=	':chart_with_upwards_trend:')

with st.sidebar:
    st.markdown("# ME607")
    st.markdown("A seguir podem ser encontrados alguns links útis referente ao trabalho aqui apresentado:")

st.title(':chart_with_upwards_trend: Trabalho Final Séries Temporais' )
st.error('Aguarde o simbolo de "Running" no canto superior para a visualização completa.')
st.subheader('Grupo: Gabriel Ukstin Talasso - 235078 ; ....')
st.markdown("## Visão geral dos dados")
st.markdown("### Alterne entre as abas para as visualizações!")

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
    fig = px.line(returns, title='Temperatura Média Diária - Vancouver', 
                  labels=({'value':'Temperatura Média', 'time':'Data'}))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

with tab2:
    fig = px.line(returns.diff(1).dropna(), title='Temperatura Média Diária - Vancouver - Diferenciada',
                  labels = {'value':'Diferença da temperatura', 'time':'Data'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

with tab3:
    st.write(data.tail(10))

c0 = st.checkbox('Mais informações sobre os dados.', help = 'Clique para saber mais sobre os dados do projeto.')

if c0:

    st.markdown('Esses dados foram coletados a partir da biblioteca meteostat, do python, que fornece informações acerca do clima de diversos pontos do mundo.')
    st.markdown('Nesse caso a cidade escolhida foi Vancouver, por conta da quantidade de dados disponíveis e ausência de falahas na coleta (como apresntadas em Campinas em São Paulo).')
    st.markdown('O foco do trabalho é predizer a temperatura média do dia seguinte, usando as temperaturas anteriores e com auxílio da variável precipitação. Outras variáveis não foram consideradas ou por se mostrarem ineficiẽntes, ou por possírem muitos valores faltantes.')

st.markdown('### Para um vislumbre da dinâmica dos dados, a seguir podemos ver os seguintes gráficos:')

tab1, tab2, tab3, tab4 = st.tabs([ "ACF - Original", 
                      "PACF - Original",
                      "ACF - Diferenciada",
                      "PACF - DIferenciada"])
with tab1:
    image = Image.open('acf.png')
    st.image(image = image)
with tab2:
    image = Image.open('pacf.png')
    st.image(image = image)

with tab3:
    image = Image.open('acfdiff.png')
    st.image(image = image)
with tab4:
    image = Image.open('pacfdiff.png')
    st.image(image = image)

st.markdown('### Modelagem')
st.markdown(' A seguir podemos ver o resulado do teste de diversos modelos, comparados atravez de uma validação cruzada de janela deslizante.')
st.markdown(' Cada modelo foi testado 30 vezes, predizedo sempre um passo a frente a raiz do erro quadratico médio (RMSE) de cada um pode ser visto na tabela abaixo')

results = pd.read_csv('comparacao_cv_30.csv').T
results = results.replace({'Seu ARIMA': 'ARIMA111', 'm5_rmse':'RMSE', 'sarima': 'SARIMA'})
results.columns = results.iloc[0]
results = results.drop(results.index[0])
results = results.set_index('Model')
results['RMSE'] = results['RMSE'].apply(lambda x: round(float(x), 3))

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.write(results)

with col3:
    st.write(' ')

c1 = st.checkbox('Mostrar mais sobre os testes.', help = 'Clique para mais informações a cerca dos experimentos para testes de modelos.')
if c1:
    st.write('O melhor modelo encontrado foi um SARIMA(1,1,3)(0,1,1)7, que desempenhou melhor nos nossos testes. '+
                'Além disso, o segundo melhor modelo foi um ARIMAX(1,0,1), usando a precipitação do dia anterior e a média da precipitação semanal como covariáveis.')
    st.markdown('OBS: Outros modelos também foram testados mas não mostrados na tabela, os apresentados são os modelos de cada tipo que tiveram melhor desempenho nos testes realizados. Modelos com período sazonal 365 dias ou não bateram os baselines ou demoravam horas para rodar, por isso foram descartados de uma anpalise diária.')

st.markdown('### Diagnóstico do modelo: SARIMA(1,1,3)(0,1,1)7')

image = Image.open('sarima_diags.png')
st.image(image = image, caption='Diagnóstico do modelo. Rejeita-se normalidade dos resíduos à 5%.')

c2 = st.checkbox('Mostrar mais sobre o diagnóstico.', help = 'Clique para mais informações a cerca do diagnóstico do modelo.')

if c2:
    st.markdown('Mesmo esse sendo o melhor modelo nos testes visualmente adequado nos gráficos. Ainda rejeitamos a normalidade dos resíduos, ou seja, o modelo ainda ainda não capturou complemente a dinâmica dos dados.')
    st.markdown('OBS: Por se tratar de um problema complexo e que envolve muitas variáveis não disponíveis, nenhum dos modelos testados obteve resíduos normais.')





