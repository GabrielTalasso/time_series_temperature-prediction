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
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


st.set_page_config('Séries Tempoais', page_icon=	':chart_with_upwards_trend:')

with st.sidebar:
    st.markdown("# ME607")
    st.markdown("A seguir podem ser encontrados alguns links útis referente ao trabalho aqui apresentado:")
    st.markdown('[Github](https://github.com/GabrielTalasso/trabalho-series)')

st.title(':chart_with_upwards_trend: Trabalho Final Séries Temporais' )
st.error('Aguarde o simbolo de "Running" no canto superior para a visualização completa.')
st.markdown('Grupo: Gabriel Ukstin Talasso - 235078 ; Tiago Henrique Silva Monteiro - 217517....')
st.markdown("## Visão geral dos dados")
st.markdown("### Alterne entre as abas para as visualizações!")

@st.cache_data # 👈 Add the caching decorator
def read_data():
        # Set time period
    start = datetime(2017, 1, 1)
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

st.markdown("#### :mostly_sunny: Visão geral -Temperaturas média diária - Vancouver")

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

st.markdown("#### :bar_chart: Médias móveis -Temperaturas média diária - Vancouver")


tab1, tab2, tab3 = st.tabs(['Média Móvel 7', 'Média Móvel 30', 'Média Móvel 300'] )
with tab1:
    fig = px.scatter(returns, trendline="rolling", title = 'Média Móvel de 3 dias da temperatura média.',
                     trendline_options=dict(window=7),
                      trendline_color_override="red")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

with tab2:
    fig = px.scatter(returns, trendline="rolling",title = 'Média Móvel de 30 dias da temperatura média.',
                      trendline_options=dict(window=30),
                       trendline_color_override="red")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


with tab3:
    fig = px.scatter(returns, trendline="rolling", title = 'Média Móvel de 300 dias da temperatura média.',
                     trendline_options=dict(window=300),
                      trendline_color_override="red")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


st.markdown("#### :umbrella_with_rain_drops: Visão geral - Precipitação diária - Vancouver")


tab1, tab2= st.tabs([ "Grafico da Série", 
                      "Matriz de correlação"])
with tab1:
    image = Image.open('images/precip.png')
    st.image(image = image, caption='Precipitação diária ao longo dos anos.')

with tab2:
    image = Image.open('images/matriz_corr.png')
    st.image(image = image, caption='Matriz de correlação entre as variáveis usadas')

c0 = st.checkbox('Mais informações sobre os dados.', help = 'Clique para saber mais sobre os dados do projeto.')

if c0:

    st.markdown('Esses dados foram coletados a partir da biblioteca meteostat, do python, que fornece informações acerca do clima de diversos pontos do mundo.')
    st.markdown('Nesse caso a cidade escolhida foi Vancouver, por conta da quantidade de dados disponíveis e ausência de falahas na coleta (como apresntadas em Campinas em São Paulo).')
    st.markdown('O foco do trabalho é predizer a temperatura média do dia seguinte, usando as temperaturas anteriores e com auxílio da variável precipitação. Outras variáveis não foram consideradas ou por se mostrarem ineficiẽntes, ou por possírem muitos valores faltantes.')

st.markdown('### :calendar: Para um vislumbre da dinâmica dos dados, a seguir podemos ver os seguintes gráficos:')

tab1, tab2, tab3, tab4 = st.tabs([ "ACF - Original", 
                      "PACF - Original",
                      "ACF - Diferenciada",
                      "PACF - DIferenciada"])
with tab1:
    image = Image.open('images/acf.png')
    st.image(image = image, caption='ACF. Teste de Ljung-Box rejeita que são não correlacionados.')
with tab2:
    image = Image.open('images/pacf.png')
    st.image(image = image)

with tab3:
    image = Image.open('images/acfdiff.png')
    st.image(image = image, caption='ACF. Teste de Ljung-Box rejeita que são não correlacionados. ')
with tab4:
    image = Image.open('images/pacfdiff.png')
    st.image(image = image)

st.markdown('### :computer: Modelagem')
st.markdown(' A seguir podemos ver o resulado do teste de diversos modelos, comparados atravez de uma validação cruzada de janela deslizante.')
st.markdown(' Cada modelo foi testado 30 vezes, predizedo sempre um passo a frente a raiz do erro quadratico médio (RMSE) de cada um pode ser visto na tabela abaixo')

results = pd.read_csv('data/comparacao_cv_30.csv').T
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

st.markdown('### :white_check_mark: Diagnóstico do modelo: SARIMA(1,1,3)(0,1,1)7')

image = Image.open('images/sarima_diags.png')
st.image(image = image, caption='Diagnóstico do modelo. Rejeita-se normalidade dos resíduos à 5%.')

c2 = st.checkbox('Mostrar mais sobre o diagnóstico.', help = 'Clique para mais informações a cerca do diagnóstico do modelo.')

if c2:
    st.markdown('Mesmo esse sendo o melhor modelo nos testes visualmente adequado nos gráficos. Ainda rejeitamos a normalidade dos resíduos, ou seja, o modelo ainda ainda não capturou complemente a dinâmica dos dados. Porém o modelo não rejeita que os resíduos são descorrelacionados, o que é um bom sinal de ajuste.')
    st.markdown('OBS: Por se tratar de um problema complexo e que envolve muitas variáveis não disponíveis, nenhum dos modelos testados obteve resíduos normais.')


st.markdown('### :clipboard: Informações ténicas sobre o modelo.')

with open('./models/model_sarima_summary.pickle', 'rb') as file:
    f = pickle.load(file)
    
st.write(f)

c3 = st.checkbox('Mais informações sobre os parâmetros do modelo.')

if c3:
    st.markdown('Acima podemos ver as as estimativas para todos os parâmetros do modelo, além disso é possível visualizar também as principais métricas de performance do modelo estudado, assim os testes comentados anteriormente.')
    st.markdown('Apesar de parte MA sasonal não se mostrar significativa, ela melhorou a desenpenho do modelo ns testes e por isso for mantida. Lembrando que esse p-valor apresentado se refere a significância da variável dado que todas outras já foram colocadas no modelo.')

