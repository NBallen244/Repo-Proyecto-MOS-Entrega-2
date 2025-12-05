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

#archivos verificacion pyomo
ruta_ver_pyomo_base = "pyomo/caso_base/verificacion_caso1.csv"
ruta_ver_pyomo_caso2 = "pyomo/caso_2/verificacion_caso2.csv"
ruta_ver_pyomo_caso3 = "pyomo/caso_3/verificacion_caso3.csv"

#archivos verificacion GA
ruta_ver_ga_base = "metaheuristica/caso_base/verificacion_metaheuristica_GA_caso_base.csv"
ruta_ver_ga_caso2 = "metaheuristica/caso_2/verificacion_metaheuristica_GA_A_caso2.csv"
ruta_ver_ga_caso3 = "metaheuristica/caso_3/verificacion_metaheuristica_GA_A_caso3.csv"

def generacion_mapa(ruta_archivo, ruta_salida):
    coordenadas_colombia = [4.5709, -74.2973]
    mapa = fo.Map(location=coordenadas_colombia, zoom_start=10)
    datos=pd.read_csv(ruta_archivo, sep=",", encoding="utf-8")
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
    mapa.save(ruta_salida)
    mapa.show_in_browser()
    
def comparacion_cargas(ruta_archivo):
    datos=pd.read_csv(ruta_archivo, sep=",", encoding="utf-8")
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
    
def comparacion_porcentual(ruta_archivo):
    datos=pd.read_csv(ruta_archivo, sep=",", encoding="utf-8")
    plt.figure(figsize=(10,6))
    datos['DemandSatisfied'] = datos['DemandSatisfied'].fillna('0-0')
    carga_total=datos['DemandSatisfied'].str.split("-").apply(lambda x: sum(map(float, x)))
    plt.pie(carga_total, labels=datos['VehicleId'], autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Carga Entregada por Vehículo')
    plt.show()

if __name__ == "__main__":
    print("Seleccione el archivo de resultados a analizar:")
    print("1. Verificación Pyomo Caso Base")
    print("2. Verificación Pyomo Caso 2")
    print("3. Verificación Pyomo Caso 3")
    print("4. Verificación Metaheurística GA Caso Base")
    print("5. Verificación Metaheurística GA Caso 2")
    print("6. Verificación Metaheurística GA Caso 3\n")
    opcion_archivo = input("Ingrese el número de la opción deseada: ")
    if opcion_archivo == '1':
        ruta_archivo = ruta_ver_pyomo_base
        ruta_salida = "pyomo/caso_base/mapa_rutas.html"
    elif opcion_archivo == '2':
        ruta_archivo = ruta_ver_pyomo_caso2
        ruta_salida = "pyomo/caso_2/mapa_rutas.html"
    elif opcion_archivo == '3':
        ruta_archivo = ruta_ver_pyomo_caso3
        ruta_salida = "pyomo/caso_3/mapa_rutas.html"
    elif opcion_archivo == '4':
        ruta_archivo = ruta_ver_ga_base
        ruta_salida = "metaheuristica/caso_base/mapa_rutas.html"
    elif opcion_archivo == '5':
        ruta_archivo = ruta_ver_ga_caso2
        ruta_salida = "metaheuristica/caso_2/mapa_rutas.html"
    elif opcion_archivo == '6':
        ruta_archivo = ruta_ver_ga_caso3
        ruta_salida = "metaheuristica/caso_3/mapa_rutas.html"
    else:
        print("Opción no válida. Saliendo del programa.")
        sys.exit(1)
    while True:
        print("Seleccione una opción:")
        print("1. Generar mapa de rutas")
        print("2. Comparar cargas entregadas vs capacidad del vehículo")
        print("3. Comparar distribución de cargas entregadas")
        print("4. Salir")
        opcion = input("Ingrese el número de la opción deseada: ")
        if opcion == '1':
            generacion_mapa(ruta_archivo, ruta_salida)
        elif opcion == '2':
            comparacion_cargas(ruta_archivo)
        elif opcion == '3':
            comparacion_porcentual(ruta_archivo)
        elif opcion == '4':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")
