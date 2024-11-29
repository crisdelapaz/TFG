import os
import numpy as np
import tensorflow as tf

directorio_red = "C:\\Users\\Cristina\\Downloads\\tfg\\datos\\red_entrenada"

def load (tipo):
    
    # carga el modelo de la red entrenada correspondiente
    
    archivo = os.path.join (directorio_red, f"red_entrenada_{tipo}.keras")
    red_entrenada = tf.keras.models.load_model(archivo)
    
    return red_entrenada

def prediction (modelo, inputs):
    
    #según las entradas del usuario, la red predice el resultado más optimo para las condiciones dichas
    inputs = np.array(inputs). reshape(1,-1)
    prediccion = modelo.predict(inputs)
    
    prediccion = prediccion.flatten()[0]
    
    print(f"la predicción es que el angulo optimo es de: {prediccion}")
    
def test():
    
    '''función principal de las predicciones. Se encarga de que el usuario 
    introduduzca que tipo de entradas va a emplear y pide la introducción de 
    ellas siempre que coincida con el tamaño deseado'''
    inputs = []
    
    print("\n\n Comprobación de la red neuronal")

    print(" LEYENDA MATRIZ 1: angulo de inclinacion, GHI, ephemeris BE, ephemeris FE, ephemeris BW, ephemeris FW, DHI/GHI, clearness index (DNI/B0), poa_front, poa_back")
    print(" LEYENDA MATRIZ 2: angulo de inclinacion, GHI, clearness index (GHI/B0), poa_front, poa_back \n" )
    
    tipo = int(input("Introduzca que opción desea: "))
    
    modelo = load (tipo)
    
    print(f"modelo {tipo} cargado")
    
    inputs_esperados = modelo.input_shape[1]
    
    print(f"introduzca los valores de entrada (se esperan {inputs_esperados}). Introduzcalos separados por comas")
    
    while True:
        
        inputs_usuario = input(f"Introduce {inputs_esperados} valores : ").strip().split(",")
        
        
        if len(inputs_usuario) != inputs_esperados:
            
            print(f"introduce {inputs_esperados} valores.")
            continue
        
        else:
            for i in inputs_usuario:
                
                valor = float (i)
                inputs.append (valor)
            break
        
    prediction(modelo, inputs)

#llamada de la función principal
test()