# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:02:33 2024

@author: miguel
"""

import pandas as pd
import numpy as np

# Ler o arquivo CSV usando o pandas
dados_csv = pd.read_csv('resultados.csv')

# Extrair os valores do DataFrame
dados = dados_csv.values

# 'Neurônios', 'Taxa de Aprendizagem', 'Erro Tolerado', 'Faixa de Inicialização', 'Máximo de Ciclos', 'Minimo Erro Quadratico', 'Tempo de Convergência'

# Encontrar o índice da linha com o valor mínimo na coluna 5
indice_mse = np.argmin(dados[:, 5])

# Encontrar o índice da linha com o valor mínimo na coluna 6
indice_tempoconverg = np.argmin(dados[:, 6])

print('MSE: ' + str(np.min(dados[:,5])))
print(dados[indice_mse,:])

print('Tempo de convergencia: ' + str(np.min(dados[:,6])))
print(dados[indice_tempoconverg,:])


