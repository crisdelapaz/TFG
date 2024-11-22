import os
import pandas as pd
import numpy as np
import pvlib as pv
from pvlib.location import Location

'''
FASE 1: TRATAMIENTO DE DATOS
'''
columna_fe = [26]
columna_fw = [28]
columna_be = [25]
columna_bw = [27]

columna_fecha = [0]
columna_GHI = [7]
#Latitude: 40º 26' 32,76" N   /   Longitude: 3º 43' 47,39" W
latitud = 40.4424333
longitud = -3.7298306
 
def csv_to_excel (directorio_origen, directorio_destino, substring, extension = ".CSV"):
    
    '''  **genera un archivo excel a partir de un archivo csv**
    la función tiene como atributos: directorio de origen y de destino,
    la cadena de caracteres que se ha de buscar (ya que no se quiere transformar todos los datos dados inicialmente)
    extensión de origen que se quiere cambiar
    '''
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
        
    for root, _, archivos in os.walk(directorio_origen):
        
        #recorre el directorio de origen y analiza los archivos con el 
        #nombre y extensión pedidos para pasar solo esos de de csv a excel
        
        for archivo in archivos:
            if archivo.endswith(extension) and substring in archivo:
                csv_ruta = os.path.join(root, archivo)
                nombre_archivo = os.path.splitext(archivo)[0]
                
                excel_ruta = os.path.join(directorio_destino, nombre_archivo + ".xlsx")
                datos = pd.read_csv(csv_ruta, delimiter = ',')
                
                datos.to_excel(excel_ruta, index = False) #genera el excel en la ruta elegida
                

def get_poa_front (excel_datos, columna_fe, columna_fw):
    
    #recoge los datos del excel de las dos columnas requeridas (para POA frOnt) 
    #POA FRONT: (FE+FW)/2
    
    fe = excel_datos.iloc[:,columna_fe[0]]
    fw = excel_datos.iloc[:,columna_fw[0]]

    poa_front = (fe+fw)/2   
    
    return poa_front

def get_poa_back (excel_datos, columna_be, columna_bw):
    
    #recoge los datos del excel de las dos columnas requeridas (para POA back) 
    #POA BACK: (BE+BW)/2   
    
    be = excel_datos.iloc[:,columna_be[0]]
    bw = excel_datos.iloc[:,columna_bw[0]]
    
    poa_back = (be+bw)/2    
    
    return poa_back

def get_clarity_index_GHI_B0 (excel_datos,columna_fecha, columna_GHI, latitud, longitud):
    
    #modifica el indice de claridad a GHI/B0
    
    fecha = pd.to_datetime(excel_datos.iloc [:, columna_fecha[0]])
    ghi = excel_datos.iloc [:, columna_GHI[0]].values

    localizacion = Location(latitude = latitud, longitude = longitud)
    pos_solar = localizacion.get_solarposition(fecha)
    
    extra_r = pv.irradiance.get_extra_radiation(fecha.dt.day_of_year)
    cos_zenith = np.cos(np.radians(pos_solar['zenith']))
    b0 = extra_r * np.abs(cos_zenith)
    
    ghi_b0 = pd.Series(ghi/b0)
    
    return ghi_b0

def split_data (matriz, nombre_archivo):
    
    '''
    separa los datos entre la parte de entrenamiento y 
    la parte de testeo y genera dos variables global por 
    cada matriz previa
    '''
    filas_entrenamiento = []
    filas_pruebas = []
    umbral = 10
    
    numero_filas = len(matriz)
    num_filas_entrenamiento = int(numero_filas * 0.8) 
    #se escoge el 80% de las filas para el entrenamiento y el resto para las pruebas
    
    if numero_filas < umbral:
        filas_entrenamiento = matriz[:num_filas_entrenamiento] 
        filas_pruebas = matriz[num_filas_entrenamiento:]     
   
    else:
        indice = np.arange(numero_filas)
        np.random.shuffle(indice)
        
        indices_filas_entrenamiento = indice[:num_filas_entrenamiento] 
        indices_filas_pruebas = indice[num_filas_entrenamiento:] 
        #escoge aleatoriamente las filas para el entrenamiento
    
        for i in indices_filas_entrenamiento:
            filas_entrenamiento.append(matriz[i])
        
        for i in indices_filas_pruebas:
            filas_pruebas.append(matriz[i])
        #las filas no escogidas para el entrenamiento se almacenan en filas_test
         
    globals()[nombre_archivo + "-training"] = filas_entrenamiento
    globals()[nombre_archivo + "-testing"] = filas_pruebas
    #return filas_entrenamiento, filas_pruebas
       
def edit_data (nombre_archivo):
   
    '''
    transforma los datos a datos tipo 
    numpy para que puedan ser procesados
    '''
    
    input_entrenamiento = []
    output_entrenamiento = []
    input_pruebas = []
    output_pruebas = []
    
    datos_entrenamiento = globals().get(nombre_archivo + "-training")
    datos_pruebas = globals().get(nombre_archivo + "-testing")

    #por cada matriz generada anteriormente de entrenamiento y prueba, 
    #se separa el input y el output 
    
    for columna in datos_entrenamiento:
        if len(columna)>1:
            input_entrenamiento.append(columna[1:])
            output_entrenamiento.append(columna[0])
        else:
            print(f"esa columna no es mayor que 1: {columna}")
            
    for columna in datos_pruebas:
        if len(columna)>1:
            input_pruebas.append(columna[1:])
            output_pruebas.append(columna[0])
        else:
            print(f"esa columna no es mayor que 1: {columna}") 
     
    input_entrenamiento = np.array(input_entrenamiento)
    output_entrenamiento = np.array(output_entrenamiento)
    input_pruebas = np.array(input_pruebas)
    output_pruebas = np.array(output_pruebas)
      
    return input_entrenamiento, output_entrenamiento, input_pruebas, output_pruebas

def read_excel (directorio_destino, columnas, get_poa = False, get_ghi_b0 = False):
    
    '''
    lee cada excel del directorio y crea una variable (array de arrays) 
    con n arrays de las n filas de cada excel  
    filtra y añade las columnas requeridas y añade después las de POA front y back.
    LLama además a la función que modifica las variables para que sean numpy y se puedan trabajar con ellas
    '''
    global_input_entrenamiento = []
    global_output_entrenamiento = []
    global_input_pruebas = []
    global_output_pruebas = []
    
    for archivo in os.listdir(directorio_destino):
        excel_ruta = os.path.join(directorio_destino, archivo)
        excel_datos = pd.read_excel(excel_ruta)
        
        columnas_seleccionadas = excel_datos.iloc[:,columnas]
        
        matriz = columnas_seleccionadas.values.tolist()
        
        if get_poa == True:
                       
            poa_front = get_poa_front(excel_datos, columna_fe, columna_fw  )
            poa_back = get_poa_back(excel_datos, columna_be, columna_bw)
        
            for i, fila in enumerate(matriz):
                fila.append(poa_front.iloc[i])
                fila.append(poa_back.iloc[i])
                
        if get_ghi_b0 == True:

            GHI_B0 = get_clarity_index_GHI_B0(excel_datos, columna_fecha, columna_GHI, latitud, longitud)
            
            for i, fila in enumerate(matriz):
                fila.append(GHI_B0.iloc[i])
                
        nombre_archivo = os.path.splitext(archivo)[0].split('-')[0]
        globals()[nombre_archivo] = matriz
        
        #se separan las "matrices" previas en entrenamiento y testeo
        split_data (matriz, nombre_archivo)
        
        input_entrenamiento, output_entrenamiento, input_pruebas, output_pruebas = edit_data(nombre_archivo)
        
        global_input_entrenamiento.append(input_entrenamiento)
        global_output_entrenamiento.append(output_entrenamiento)
        global_input_pruebas.append(input_pruebas)
        global_output_pruebas.append(output_pruebas)
        
    global_input_entrenamiento = np.vstack(global_input_entrenamiento)
    global_output_entrenamiento = np.hstack(global_output_entrenamiento)
    global_input_pruebas = np.vstack(global_input_pruebas)
    global_output_pruebas = np.hstack(global_output_pruebas)
        
    return global_input_entrenamiento, global_output_entrenamiento, global_input_pruebas, global_output_pruebas
