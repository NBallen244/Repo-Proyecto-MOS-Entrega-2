import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import folium as fo
from folium.plugins import MarkerCluster
# Add the directory containing cargaBase.py to Python's search path
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from carga_datos.cargaDatos import cargar_datos_base
colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

def generacion_mapa():
    coordenadas_colombia = [4.5709, -74.2973]
    mapa = fo.Map(location=coordenadas_colombia, zoom_start=10)
    datos=pd.read_csv("caso_base/verificacion_caso1.csv", sep=",", encoding="utf-8")
    detalles=cargar_datos_base()
    clientes=detalles[0]
    depositos=detalles[1]
    marker_cluster = MarkerCluster().add_to(mapa)
    for index, fila in datos.iterrows():
        secuencia=fila['RouteSequence'].split("-")
        ruta=[]
        color=colors[index % len(colors)]
        orden=0
        for nodo in secuencia:
            if "D" in nodo:
                info_nodo=depositos.loc[depositos['StandardizedID'] == nodo].iloc[0]
            else:
                info_nodo=clientes.loc[clientes['StandardizedID'] == nodo].iloc[0]
            marker_color=color
            location=[info_nodo['Latitude'], info_nodo['Longitude']]
            ruta.append(location)
            fo.Marker(location=location, popup=fo.Popup(f"{nodo} en ruta del vehiculo {fila['VehicleId']} orden {orden}", max_width=300), icon=fo.Icon(color=marker_color)).add_to(marker_cluster)
            orden += 1
        fo.PolyLine(ruta, color=color, weight=2.5, opacity=1).add_to(mapa)
    mapa.save("caso_base/mapa_rutas.html")
    
def comparacion_cargas():
    datos=pd.read_csv("caso_base/verificacion_caso1.csv", sep=",", encoding="utf-8")
    datos['DemandSatisfied'] = datos['DemandSatisfied'].fillna('0-0')
    demandas = datos['DemandSatisfied'].str.split("-").tolist()
    cargas_totales = [sum(float(demand) for demand in demanda) for demanda in demandas]
    desperdicio = datos['LoadCap'] - pd.Series(cargas_totales)
    porcentaje_desperdicio = (desperdicio / datos['LoadCap']) * 100
    
    plt.figure(figsize=(10,6))
    plt.plot(datos['VehicleId'], cargas_totales, marker='o', label='Carga Entregada (kg)')
    plt.plot(datos['VehicleId'], datos['LoadCap'], marker='s', label='Capacidad del Vehículo (kg)')
    plt.bar(datos['VehicleId'], porcentaje_desperdicio, alpha=0.2, color='red', label='Porcentaje de desperdicio (%)')
    plt.xlabel('ID del Vehículo')
    plt.ylabel('Carga (kg)/Porcentaje (%)')
    plt.title('Comparación de Carga Entregada vs Capacidad del Vehículo')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def comparacion_porcentual():
    datos=pd.read_csv("caso_base/verificacion_caso1.csv", sep=",", encoding="utf-8")
    plt.figure(figsize=(10,6))
    datos['DemandSatisfied'] = datos['DemandSatisfied'].fillna('0-0')
    carga_total=datos['DemandSatisfied'].str.split("-").apply(lambda x: sum(map(float, x)))
    plt.pie(carga_total, labels=datos['VehicleId'], autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Carga Entregada por Vehículo')
    plt.show()

if __name__ == "__main__":
    while True:
        print("Seleccione una opción:")
        print("1. Generar mapa de rutas")
        print("2. Comparar cargas entregadas vs capacidad del vehículo")
        print("3. Comparar distribución de cargas entregadas")
        print("4. Salir")
        opcion = input("Ingrese el número de la opción deseada: ")
        if opcion == '1':
            generacion_mapa()
        elif opcion == '2':
            comparacion_cargas()
        elif opcion == '3':
            comparacion_porcentual()
        elif opcion == '4':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")
