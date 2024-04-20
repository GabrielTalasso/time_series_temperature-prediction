## Time Series Modeling and Automatizating 

Authors: Gabriel Ukstin Talasso, Marcos José Grosso Filho e Tiago Henrique Silva Monteiro.

#### This repository contains the codes for a complete workflow on a time series project: analysis, modeling and the diagnostics of temporal data.

<img src="https://github.com/GabrielTalasso/trabalho-series/assets/75808460/201dbb79-04fa-429f-b228-24f3adb71dbe" align="center" width="600">



The data was obtained using the "meteostat" python library, that provides data about the climate on several cities of the world. In this project we use the Vancouver climate data, because this city there is no missing data in the period.

The main focus of this project is predict the average temperature of the next day, using the previous information like temperatures and precipitation. Other variables were considered but proved to be inefficient, or because they had many missing values.

Futhermore, this project contains an `streamli_app.py` ([link](https://gabrieltalasso-trabalho-series-streamlit-app-vug9p4.streamlit.app/)) that shows all resuts of the project. And the code to automate the sending of emails with the daily analyzes (the second part of the project) can be found at [this Github](https://github.com/Marcosgrosso/automation_series). The model used on this this automatic forecast is in `predict_model.py`.

The choice of the model was made in `experiments.py` using a Time Series Cross Validation (`tscv.py`) and validate in `diagnostics.py`.





