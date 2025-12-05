import pandas as pd
#tomando la ruta de los archivos con el directorio de trabajo siendo la carpeta raiz del proyecto
ruta_archivos_base = "carga_datos/Proyecto_Caso_Base/"
ruta_archivos_caso2 = "carga_datos/project_a/Proyecto_A_Caso2/"
ruta_archivos_caso3 = "carga_datos/project_a/Proyecto_A_Caso3/"

columnas_clientes=["ClientID", "StandardizedID", "LocationID", "Latitude", "Longitude", "Demand"]

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
    parametros= pd.read_csv(ruta_archivos_caso2+"parameters_urban.csv", sep=",", encoding="latin1")
    # Cargar vehiculos
    vehiculos= pd.read_csv(ruta_archivos_caso2+"vehicles.csv", sep=",", encoding="latin1")

    return clientes, depositos, parametros, vehiculos


def cargar_datos_caso3():
    # Cargar el archivo clients.csv
    clientes= pd.read_csv(ruta_archivos_caso3+"clients.csv", sep=",", encoding="latin1")
    # Cargar el archivo depots.csv
    depositos= pd.read_csv(ruta_archivos_caso3 +"depots.csv", sep=",", encoding="latin1")
    # Cargar el archivo parameters_base.csv
    parametros= pd.read_csv(ruta_archivos_caso3+"parameters_urban.csv", sep=",", encoding="latin1")
    # Cargar vehiculos
    vehiculos= pd.read_csv(ruta_archivos_caso3 +"vehicles.csv", sep=",", encoding="latin1")
    return clientes, depositos, parametros, vehiculos

if __name__ == "__main__":
    # Prueba de carga de datos
    clientes_base, depositos_base, parametros_base, vehiculos_base = cargar_datos_base()
    print("Datos del caso base cargados correctamente.")
    clientes_caso2, depositos_caso2, parametros_caso2, vehiculos_caso2 = cargar_datos_caso2()
    print("Datos del caso 2 cargados correctamente.")
    clientes_caso3, depositos_caso3, parametros_caso3, vehiculos_caso3 = cargar_datos_caso3()
    print("Datos del caso 3 cargados correctamente.")
