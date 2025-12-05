import os
import sys
import pandas as pd

# Add the directory containing cargaBase.py to Python's search path
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from herramientas_compartidas.distancia import osrm_distance
from carga_datos.cargaDatos import cargar_datos_caso3 as cargar_datos_base

def gen_csv_distancia_tiempo (nom_archivo, clientes, depositos):
    # Genera un archivo CSV con las distancias y tiempos entre nodos usando OSRM para cualquier caso
    dict_archivo={
        "FromID":[],
        "ToID":[],
        "Distance_km":[],
        "Time_min":[]
    }
    for i, fila_i in clientes.iterrows():
        for j, fila_j in clientes.iterrows():
            if i != j:
                dist, time = osrm_distance(
                    (fila_i["Longitude"], fila_i["Latitude"]),
                    (fila_j["Longitude"], fila_j["Latitude"])
                )
                dict_archivo["FromID"].append(fila_i["StandardizedID"])
                dict_archivo["ToID"].append(fila_j["StandardizedID"])
                dict_archivo["Distance_km"].append(dist)
                dict_archivo["Time_min"].append(time)
    for i, fila_i in depositos.iterrows():
        for j, fila_j in clientes.iterrows():
            dist, time = osrm_distance(
                (fila_i["Longitude"], fila_i["Latitude"]),
                (fila_j["Longitude"], fila_j["Latitude"])
            )
            dict_archivo["FromID"].append(fila_i["StandardizedID"])
            dict_archivo["ToID"].append(fila_j["StandardizedID"])
            dict_archivo["Distance_km"].append(dist)
            dict_archivo["Time_min"].append(time)
    df_distancias = pd.DataFrame(dict_archivo)
    df_distancias.to_csv(nom_archivo, index=False)

if __name__ == "__main__":
    ruta_archivo="pyomo/caso_base/matriz.csv"
    clientes, depositos = cargar_datos_base()[0:2]
    gen_csv_distancia_tiempo(ruta_archivo, clientes, depositos)