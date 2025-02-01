import os
import pandas as pd
import numpy as np
import re

import tensorflow as tf
import matplotlib.pyplot as plt

'''
FASE DOS: CREACIÓN Y ENTRENAMIENTO DE LA RED
'''

def plot_red_neuronal (historial, titulo):
    
    #grafica los resultados de la red: añade leyenda, titulo, los dos tipos de pérdida
    
    plt.xlabel("iteración")
    plt.ylabel("pérdida")
    plt.plot(historial.history["loss"], label = "Pérdida de entrenamiento")
    plt.plot(historial.history["val_loss"], label = "Pérdida de testeo")
    
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()    
    
def create_model (in_entrenamiento, in_pruebas, numero_red):
    
    '''crea el modelo de red neuronal de tipo ANN con sus respectivas capas intermedias'''
    modelo = tf.keras.Sequential()

    inputs = in_entrenamiento.shape[1]
    entrada = tf.keras.layers.Input(shape=(inputs,))
    
    modelo.add(entrada)
    n = 0
    
    while ( n < 3 ):
        #selecciona si añadir o no capas
        opcion = int(input("introduzca una de las opciones: 1) añadir capa 0) no añadir capa: "))
        
        if n == 0 and opcion == 0:
            print("no se añadirá capas intermedias")
            capas = 0
        
        elif opcion == 1:
            
            if n == 0:
                #seleccion de numero de neuronas
                neurona = int(input("introduzca numero de neuronas: 1) 4 2) 8 3) 16 4) 32: "))
                
                if neurona == 1:
                    numero = 4
                elif neurona == 2:
                    numero = 8
                elif neurona == 3:
                    numero = 16
                else:
                    numero = 32
                
                #seleccion de la funcion de activacion
                tipo = int(input("introduzca funcion de activacion: 1) tanh 2) sigmoid 3) relu: "))
                
                if tipo == 1:
                    funcion = 'tanh'
                elif tipo == 2:
                    funcion = 'sigmoid'
                else:
                    funcion = 'relu'
                
                print(f"capa añadida de {numero} neuronas con función {funcion} ")
            else:
                print("se añadirá una capa igual a la anterior")
                       
            modelo.add(tf.keras.layers.Dense(numero, activation = funcion))
            
            capas = n + 1
            n+=1
            
        else:
            print("no se añadirá capa intermedia")
            break
    
        if capas == 0:
            titulo = (f"{numero_red}- ANN con 0 Capas intermedias ")
            break                
        else:
            titulo = (f"{numero_red}- ANN {capas} Capa de {numero} Neuronas con función {funcion.upper()} ")
            
    print (titulo)    
    salida = tf.keras.layers.Dense(1, activation='linear' )
    modelo.add (salida)
    
    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss ='mean_squared_error'
        )    

    return modelo, titulo, in_entrenamiento, in_pruebas
    
def training (in_entrenamiento, out_entrenamiento, in_pruebas, out_pruebas, numero_red): 
    
    '''
    se crea el modelo de red neuronal que se va desarrollar (Secuencial)
    y el tipo de relaciones entre capas de neuronas (densa - para que cada neurona
    se relacione con todas las neuronas de la siguiente capa)
    '''
    modelo, titulo, in_entrenamiento, in_pruebas = create_model(in_entrenamiento, in_pruebas, numero_red)
    
    print("\n Comenzando el entrenamiento...")
    
    historial = modelo.fit(in_entrenamiento, out_entrenamiento,
                          validation_data=(in_pruebas, out_pruebas),
                          epochs=31, 
                          batch_size = 32, 
                          verbose= True)

    print("... Entrenado!!")
    
    plot_red_neuronal(historial, titulo)

    return modelo, titulo
'''
FIN ENTRENAMIENTO RED
'''
'''
FASE 3: COMPROBACIÓN DE RESULTADOS DE LA RED
'''
def testing(inputs, outputs, modelo_entrenado, nombre, directorio, nombre_columna, nombre_output):
    
    '''
    genera un excel con los inputs,outputs, el error cuadratico
    y la prediccion. Se guarda en un directorio designado
    '''
    
    nombre_simplificado = '_'.join(re.findall(r'[A-Z0-9]+', nombre))
    nombre_excel = f"{nombre_simplificado}.xlsx"

    
    prediccion = modelo_entrenado.predict(inputs).flatten()
    e_grados = prediccion - outputs
    e_cuadratico = np.square(prediccion - outputs)

    #se modifican los inputs, outputs, predicciones y errores a dataframe para pasarlos despues al excel
    df_inputs = pd.DataFrame(inputs, columns= nombre_columna)
    df_outputs = pd.DataFrame(outputs, columns = nombre_output)
    df_prediccion = pd.DataFrame(prediccion, columns = ['Prediccion'])
    df_e_grados = pd.DataFrame(e_grados, columns = ['Desviación de grados'])
    df_e_cuadratico = pd.DataFrame(e_cuadratico, columns = ['Errores cuadraticos'])
    
    #añade los inputs, outputs y de los demás datos al excel que se va a generar
    datos_excels = pd.concat([
        df_inputs, df_outputs, df_prediccion, 
        df_e_grados, df_e_cuadratico], axis=1)
    
    if not os.path.exists(directorio):
        os.makedirs(directorio)  
        
    ruta = os.path.join(directorio, nombre_excel)
    
    r=1
    while os.path.exists(ruta):
        #se verifica que exista ya el archivo
        nombre_excel = f"{nombre_simplificado}_v{r}.xlsx"
        ruta = os.path.join(directorio, nombre_excel)
        r += 1

    datos_excels.to_excel(ruta, index = False)
    
    print("Comprobación!! Revisa el directorio para revisar los resultados")
'''
FIN RESULTADOS RED
'''




