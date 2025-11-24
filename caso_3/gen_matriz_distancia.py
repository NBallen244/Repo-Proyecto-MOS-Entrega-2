import os
import sys
import pandas as pd

# Agregar el directorio que contiene cargaDatos.py al path de Python
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from herramientas_compartidas.distancia import osrm_distance
from carga_datos.cargaDatos import cargar_datos_caso3

def gen_csv_distancia_tiempo_caso3(nom_archivo, clientes, depositos, estaciones, peajes):
    """
    Genera un archivo CSV con las distancias y tiempos entre
    todos los nodos del caso 3 (clientes, depósito, estaciones, peajes).
    Usa la API OSRM y hace fallback a Haversine.
    """
    # Unir todos los nodos en un solo DataFrame
    nodos = pd.concat([clientes, depositos, estaciones, peajes], ignore_index=True)

    dict_archivo = {
        "FromID": [],
        "ToID": [],
        "Distance_km": [],
        "Time_min": []
    }

    for i, fila_i in nodos.iterrows():
        for j, fila_j in nodos.iterrows():
            if i != j:
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
    print(f"Matriz de distancia-tiempo guardada en: {nom_archivo}")

if __name__ == "__main__":
    ruta_archivo = "caso_3/matriz.csv"
    clientes, depositos, parametros, vehiculos, estaciones, peajes = cargar_datos_caso3()
    gen_csv_distancia_tiempo_caso3(ruta_archivo, clientes, depositos, estaciones, peajes)
