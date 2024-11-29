import os

from tratamiento_datos import rd_excel, csv_to_excel
from red_neuronal import training, testing

#variables y llamadas a funciones 
directorio_origen = "C:\\Users\\Cristina\\Downloads\\tfg\\datos\\datos_originales"
directorio_destino = "C:\\Users\\Cristina\\Downloads\\tfg\\datos\\datos_convertidos"
directorio_datos_generados = "C:\\Users\\Cristina\\Downloads\\tfg\\datos\\datos_generados"
substring = "training_data"

directorio_red = "C:\\Users\\Cristina\\Downloads\\tfg\\datos\\red_entrenada"


def red_creation(columnas, nombre_columna, lector, numero_red, get_poa, get_ghi_bo):
    
    #después de ser seleccionadas las entradas, se procede a la separación 
    #entre inputs y outputs de entrenamiento y prueba; la creación de la red y sus posteriores pruebas
    nombre_output = ['Angulo óptimo']
    
    #fase 1
    #lee el excel y lo filtra en función de lo pedido, y genera los inputs y outputs que van a ser generados
    input_entrenamiento, output_entrenamiento, input_prueba, output_prueba = lector (directorio_destino, columnas, get_poa, get_ghi_bo)
    
    #fase 2
    #se crea la red y se entrena
    modelo_entrenado, nombre_opcion = training(input_entrenamiento, output_entrenamiento, input_prueba, output_prueba, numero_red)
    
    #fase 3
    #se prueba el funcionamiento de la red 
    testing(input_prueba, output_prueba, modelo_entrenado, nombre_opcion, directorio_datos_generados, nombre_columna, nombre_output)
    
    return modelo_entrenado

def input_set_selection():
    
    ''' se selecciona con que entradas se va a trabajar.'''
    
    #crea el directorio deseado para almacenar los archivos correspondientes y no genera error si ya existía de antes el directorio
    os.makedirs(directorio_destino, exist_ok = True)
    os.makedirs(directorio_datos_generados, exist_ok = True)
    os.makedirs(directorio_red, exist_ok = True)
    #crea el excel a partir del csv
    csv_to_excel(directorio_origen, directorio_destino, substring)
    
    
    while True:
        
        print(" LEYENDA MATRIZ 1: optimal angle, tilt, GHI, ephemeris BE, ephemeris FE, ephemeris BW, ephemeris FW, DHI/GHI, clearness index (DNI/B0), poa_front, poa_back")
        print(" LEYENDA MATRIZ 2: optimal angle, tilt, GHI, clearness index (GHI/B0), poa_front, poa_back \n" )
        print(" Cualquier otro numero finalizará el bucle")
        
        seleccion = int(input("Introduzca que entradas desea: "))
        
        if seleccion == 1:
            '''PRIMERA RED'''
            
            print("ha elegido la primera red")
            
            columnas = [5, 6, 7, 45, 46, 47, 48, 53, 56]
    
            nombre_columna = ['Ángulo del modulo', 'GHI', 'efemerides BE',
                              'efemerides FE', 'efemerides BW', 
                              'efemerides FW', 'DHI/GHI',
                              'indice de claridad (DNI/B0)', 'POA_front', 'POA_back']
            
            modelo_entrenado = red_creation(columnas, nombre_columna, rd_excel, seleccion, True, False)
    
            """ LEYENDA MATRIZ 1
            0 optimal angle
            1 tilt
            2 GHI
            3 ephemeris BE
            4 ephemeris FE
            5 ephemeris BW
            6 ephemeris FW
            7 DHI/GHI
            8 clearness index (DNI/B0)
            9 poa_front
            10 poa_back
            """
        
        elif seleccion == 2:
            '''SEGUNDA RED'''
            
            print("ha elegido la segunda red")
            
            columnas = [5, 6, 7]
    
            nombre_columna = ['Ángulo del modulo', 'GHI', 'POA_front', 'POA_back',
                              'indice de claridad (GHI/B0)']
            
            modelo_entrenado = red_creation(columnas, nombre_columna, rd_excel, seleccion, True, True )
    
            """ LEYENDA MATRIZ 2
            0 optimal angle
            1 tilt
            2 GHI
            3 clearness index (GHI/B0)
            4 poa_front
            5 poa_back
            """
        
        else:
            print ("Fin de bucle")
            break
        #guarda el modelo entrenado siendo un archivo .keras
        export (modelo_entrenado, directorio_red, seleccion)

def export (modelo, directorio_red, seleccion):
    
    '''guarda el modelo entrenado en una ruta especificada y lo separa en función de las entradas escogidas'''
    if not os.path.exists(directorio_red):
        os.makedirs(directorio_red)
    
    if seleccion == 1:
        arc_modelo = os.path.join(directorio_red, 'red_entrenada_1.keras')
        
    elif seleccion == 2:
        arc_modelo = os.path.join(directorio_red, 'red_entrenada_2.keras')
        
    modelo.save(arc_modelo)
'''
FIN TRATAMIENTO DE DATOS
'''

input_set_selection()

""" leyenda EXCEL

0    DateTime
1    optimal power
2    optimal power non bifacial
3    ephemeris power
4    ephemeris power non bifacial
5    optimal angle
6    tilt
7    GHI
8    dGHI
9    BE-exterior
10    BE-mid-exterior
11    BE-mid-interior
12    BE-interior
13    FE-exterior
14    FE-mid-exterior
15    FE-mid-interior
16    FE-interior
17    BW-exterior
18    BW-mid-exterior
19    BW-mid-interior
20    BW-interior
21    FW-exterior
22    FW-mid-exterior
23    FW-mid-interior
24    FW-interior
25    BE
26    FE
27    BW
28    FW
29    ephemeris BE-exterior
30    ephemeris BE-mid-exterior
31    ephemeris BE-mid-interior
32    ephemeris BE-interior
33    ephemeris FE-exterior
34    ephemeris FE-mid-exterior
35    ephemeris FE-mid-interior
36    ephemeris FE-interior
37    ephemeris BW-exterior
38    ephemeris BW-mid-exterior
39    ephemeris BW-mid-interior
40    ephemeris BW-interior
41    ephemeris FW-exterior
42    ephemeris FW-mid-exterior
43    ephemeris FW-mid-interior
44    ephemeris FW-interior
45    ephemeris BE
46    ephemeris FE
47    ephemeris BW
48    ephemeris FW
49    Optimal Bifacial gains
50    Optimal Bifacial gains irr
51    Ephemeris Bifacial gains
52    Ephemeris Bifacial gains irr
53    DHI/GHI
54    Wspeed
55    Temp
56    clearness index

"""
