import pandas as pd
#tomando la ruta de los archivos con el directorio de trabajo siendo la carpeta raiz del proyecto
ruta_archivos_base = "carga_datos/Proyecto_Caso_Base/"
ruta_archivos_caso2 = "carga_datos/project_c/Proyecto_C_Caso2/"
ruta_archivos_caso3 = "carga_datos/project_c/Proyecto_C_Caso3/"

def cargar_datos_base():
    # Cargar el archivo clients.csv
    clientes= pd.read_csv(ruta_archivos_base+"clients.csv", sep=",", encoding="latin1")
    # Cargar el archivo depots.csv
    depositos= pd.read_csv(ruta_archivos_base+"depots.csv", sep=",", encoding="latin1")
    # Cargar el archivo parameters_base.csv
    parametros= pd.read_csv(ruta_archivos_base+"parameters_base.csv", sep=",", encoding="latin1")
    # Cargar vehiculos
    vehiculos= pd.read_csv(ruta_archivos_base+"vehicles.csv", sep=",", encoding="latin1")

    return clientes, depositos, parametros, vehiculos


def cargar_datos_caso2():
    # Cargar el archivo clients.csv
    clientes= pd.read_csv(ruta_archivos_caso2+"clients.csv", sep=",", encoding="latin1")
    # Cargar el archivo depots.csv
    depositos= pd.read_csv(ruta_archivos_caso2+"depots.csv", sep=",", encoding="latin1")
    # Cargar el archivo parameters_base.csv
    parametros= pd.read_csv(ruta_archivos_caso2+"parameters_national.csv", sep=",", encoding="latin1")
    # Cargar vehiculos
    vehiculos= pd.read_csv(ruta_archivos_caso2+"vehicles.csv", sep=",", encoding="latin1")
    # Cargar estaciones de recarga
    estaciones= pd.read_csv(ruta_archivos_caso2+"stations.csv", sep=",", encoding="latin1")

    return clientes, depositos, parametros, vehiculos, estaciones


def cargar_datos_caso3():
    # Cargar el archivo clients.csv
    clientes= pd.read_csv(ruta_archivos_caso3+"clients.csv", sep=",", encoding="latin1")
    # Cargar el archivo depots.csv
    depositos= pd.read_csv(ruta_archivos_caso3 +"depots.csv", sep=",", encoding="latin1")
    # Cargar el archivo parameters_base.csv
    parametros= pd.read_csv(ruta_archivos_caso3+"parameters_national.csv", sep=",", encoding="latin1")
    # Cargar vehiculos
    vehiculos= pd.read_csv(ruta_archivos_caso3 +"vehicles.csv", sep=",", encoding="latin1")
    # Cargar estaciones de recarga
    estaciones= pd.read_csv(ruta_archivos_caso3+"stations.csv", sep=",", encoding="latin1")
    # Cargar peajes
    peajes= pd.read_csv(ruta_archivos_caso3+"tolls.csv", sep=",", encoding="latin1")
    return clientes, depositos, parametros, vehiculos, estaciones, peajes