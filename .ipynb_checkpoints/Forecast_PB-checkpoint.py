import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from tqdm import tqdm 
import itertools
 
 
class ForecastPB():
 #Classe de forecast usando algoritmo prophet da Meta recebendo dados agregados mensalmente
 #Feito para dados do Paraná Banco
    def __init__(self, df):
    	#Aqui é preciso passar um dataframe como parâmetro para iniciar a classe
        #O dataframe precisa de duas colunas, uma de data nomeada ds e outra com a serie em si nomeada y  
       
        self.df = df
        return
        
    
    def tunar_modelo(self):
       	#Funcao para tunar o modelo com um grid de parametros do prophet
        #OBS: nao quis criar um grid muito grande pela demora do tunning 
    
    	#pequeno grid de hiperparametros do prophet
        grid_param = {  
            'changepoint_prior_scale': [0.1, 0.5, 0.75, 1.0], #o quao sensivel o modelo fica a tendencias
            'seasonality_prior_scale': [0.0001, 0.001, 0.01, 0.1], #o quao sensivel o modelo fica a sazonalidades
            'seasonality_mode': ['additive', 'multiplicative'], #tipo de sazonalidade
        }
        # Gerando todas as possiveis combinacoes de parametros
        combinacoes = [dict(zip(grid_param.keys(), v)) for v in itertools.product(*grid_param.values())]
        mapes = []  # lista para guardar erro medio percentual de todos os modelos
        # Usando cross validation para avaliar todos os parametros:
        for params in tqdm(combinacoes):
            m = Prophet(**params).fit(self.df)  # Fit do modelo no df para cada combinacao
            df_cv = cross_validation(model=m,
            												 initial='365.25 days', #tempo de treino na serie 1 ano
                                     period='30 days', # distancia entre cortes
                                     horizon = '90 days', #horizonte de previsao
                                     parallel='processes')
                                     
            df_p = performance_metrics(df_cv, rolling_window=1) #metricas de performance
            mapes.append(df_p['mape'].values[0]) #salvando MAPE
            
        # Encontrando o melhor resultado 
        resultados = pd.DataFrame(combinacoes)
        resultados['mape'] = mapes
        self.melhores_parametros = combinacoes[np.argmin(mapes)]
        
        
        print(resultados.sort_values('mape'))
        print('-'*25)
        print(f'Melhores parametros:{self.melhores_parametros}')
            
        # Modelo final
        self.m = Prophet(**self.melhores_parametros).fit(self.df)  #Fit no melhor modelo
        
        #cross validation novamente para metricas do melhor modelo
        self.df_cv = cross_validation(model=self.m,
            												 initial='365.25 days', #tempo de treino na serie 1 ano
                                     period='30 days', # distancia entre cortes
                                     horizon = '90 days', #horizonte de previsao
                                     parallel='processes')
                                     
        self.df_p = performance_metrics(self.df_cv, rolling_window=1)
        
        return
        
    def predict(self, meses):
   # Predicao com modelo fitado e tunado
        #criando dataframe de tempo futuro
        future = self.m.make_future_dataframe(periods=meses,freq='ME')
        
        #fazendo predicao no dataframe de tempo futuro
        self.forecast = self.m.predict(future)
        
        #plot do forecast
        self.m.plot(self.forecast)
        
        return
         
        