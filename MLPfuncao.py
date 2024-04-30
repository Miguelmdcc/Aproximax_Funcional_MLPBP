import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

print("\x1b[2J\x1b[1;1H") 

# sen(X)*sen(2X).
entradas = 1
neur = 0
alfa = 0
errotolerado = 0
# Definir faixas de valores para os parâmetros
neurons_range = [50, 100, 200]
learning_rate_range = [0.001, 0.005, 0.01]
error_range = [0.01, 0.02, 0.05]
weight_init_ranges = [(0, 0.1), (-0.5, 0.5), (-1, 1)]
max_cycles_range = [1000, 2000, 3000]

# Listas para armazenar os resultados
results = []

listaciclo = []
listaerro = []
xmin = -1 # Limite inferior da funo.
xmax = 1 # Limite superior da funo.
npontos = 50 # Nmero de pontos igualmente espaados.

#Gerando o arquivo de entradas
x_orig = np.linspace(xmin,xmax,npontos) # Criao dos pontos igualmente espaados.
x = np.zeros((npontos,1))
for i in range(npontos):
    x[i][0]=x_orig[i] # Entradas prontas para a RNA.
#End For

(amostras,vsai) = np.shape(x) # 50 amostras e 1 sada.

t_orig = (np.sin(x))*(np.sin(2*x)) # Target "puro".
t = np.zeros((1,amostras))
for i in range(amostras):
    t[0][i]=t_orig[i] # Target pronto para a RNA.
#End For

(vsai,amostras) = np.shape(t) # [50][1]

# Loop sobre todas as combinações de parâmetros
for neurons in neurons_range:
    neur = neurons
    for learning_rate in learning_rate_range:
        alfa = learning_rate
        for error_tolerance in error_range:
            errotolerado = error_tolerance
            for weight_init_range in weight_init_ranges:
                for max_cycles in max_cycles_range:
                    
                    # Gerando os pesos sinpticos aleatoriamente.
                    v = np.random.uniform(weight_init_range[0], weight_init_range[1], size=(entradas,neur))
                    
                    v0 = np.random.uniform(weight_init_range[0], weight_init_range[1], size=(1,neur))
                    
                    w = np.random.uniform(weight_init_range[0], weight_init_range[1], size=(neur,vsai))
                    
                    w0 = np.random.uniform(weight_init_range[0], weight_init_range[1], size=(1,vsai))
                    
                    # Matrizes de atualizao de pesos e valores de saida da rede.
                    vnovo = np.zeros((entradas,neur))
                    v0novo = np.zeros((1,neur))
                    wnovo = np.zeros((neur,vsai))
                    w0novo = np.zeros((1,vsai))
                    zin_j = np.zeros((1,neur))
                    z_j = np.zeros((1,neur))
                    deltinha_k = np.zeros((vsai,1))
                    deltaw0 = np.zeros((vsai,1))
                    deltinha_j = np.zeros((1,neur))
                    x_linhaTransp = np.zeros((1,entradas))
                    y_transp = np.zeros((vsai,1))
                    t_transp = np.zeros((vsai,1))
                    deltinha_jTransp = np.zeros((neur,1))
                    ciclo = 0
                    errototal=1
                    mse = 0
                    tempoconvergencia = 0
                    start_time = time.time()
                    
                    while errotolerado < errototal and ciclo < max_cycles:
                        errototal=0
                        for padrao in range(amostras):
                            for j in range(neur):
                                zin_j[0][j] = np.dot(x[padrao,:],v[:,j]) + v0[0][j]
                            #End For
                     
                            z_j = np.tanh(zin_j) # Funo de ativao.
                            
                            yin = np.dot(z_j,w) + w0 # Saída pura.
                            y = np.tanh(yin) # Sada lquida.
                         
                            for m in range(vsai):
                                y_transp[m][0] = y[0][m] 
                            #End For
                            
                            for m in range(vsai):
                                t_transp[m][0]=t[0][padrao]
                            #End For
                            errototal = errototal+(0.5*(np.sum(((t_transp-y_transp)**2))))
                            
                            # Busca das matrizes para atualizao dos pesos.
                            deltinha_k = (t_transp - y_transp)*(1 + y_transp)*(1 - y_transp)
                            deltaw = alfa*(np.dot(deltinha_k,z_j))
                            deltaw0 = alfa * deltinha_k
                            deltinhain_j = np.dot(np.transpose(deltinha_k),np.transpose(w))
                            deltinha_j = deltinhain_j * (1 + z_j) * (1 - z_j)
                            for m in range(neur):
                                deltinha_jTransp[m][0] = deltinha_j[0][m]
                            #End For
                            for k in range(entradas):
                                x_linhaTransp[0][k] = x[padrao][k]
                            #End For
                            
                            deltav = alfa * np.dot(deltinha_jTransp,x_linhaTransp)
                            deltav0 = alfa * deltinha_j
                            
                            #Realizando as atualizaes de pesos e bias.
                            vnovo = v + np.transpose(deltav)
                            v0novo = v0 + np.transpose(deltav0)
                            
                            wnovo = w + np.transpose(deltaw)
                            w0novo = w0 + np.transpose(deltaw0)
                            
                            # Preparo para o prximo lao.
                            v =vnovo
                            v0 = v0novo
                            w = wnovo
                            w0 = w0novo
                        #End: for padrao in range(amostras):   
                        ciclo = ciclo+1
                        listaciclo.append(ciclo)
                        listaerro.append(errototal)
                        print('Ciclo\t Erro')
                        print(ciclo,'\t',errototal)
                    
                        # Comparao target e y.
                        zin2_j = np.zeros((1,neur))
                        z2_j = np.zeros((1,neur))
                        t_teste = np.zeros((amostras,1))
                        
                        for i in range(amostras):
                            for j in range(neur):
                                zin2_j[0][j] = np.dot(x[i,:],v[:,j])+v0[0][j]
                                z2_j = np.tanh(zin2_j)
                            #End For
                            yin2 = np.dot(z2_j,w) + w0
                            y2 = np.tanh(yin2)
                            t_teste[i][0] = y2
                        #End For
                        mse = np.mean((t_teste - t)**2)
                    #End While
                    end_time = time.time()
                    tempoconvergencia = end_time - start_time

                    plt.plot(x,t_orig,color='red')
                    plt.plot(x,t_teste,color='blue')
                    plt.show() 

                    results.append((neurons, learning_rate, error_tolerance, weight_init_range, max_cycles, mse,tempoconvergencia))
                    
# Convertendo a lista de resultados em um DataFrame do pandas
df = pd.DataFrame(results, columns=['Neurônios', 'Taxa de Aprendizagem', 'Erro Tolerado', 'Faixa de Inicialização', 'Máximo de Ciclos', 'Minimo Erro Quadratico', 'Tempo de Convergência'])
# Salvando o DataFrame em um arquivo CSV
df.to_csv('resultados.csv', index=False)
