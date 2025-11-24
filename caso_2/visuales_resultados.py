# =====================================================
# visuales_resultados.py — LogistiCo Proyecto C: Visualización de resultados Caso 2
# =====================================================

import os
import sys
import pandas as pd
import folium as fo
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# Importación de datos base
# ----------------------------------------------------
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)
from carga_datos.cargaDatos import cargar_datos_caso2 as cargar_datos

# Directorio de salida
OUTPUT_DIR = "caso_2/visualizaciones_caso2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

colors = [
    'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige',
    'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'gray', 'black'
]

# ----------------------------------------------------
# 1. Mapa con rutas nacionales y estaciones
# ----------------------------------------------------
def mapa_rutas():
    verificacion = pd.read_csv("caso_2/verificacion_caso2.csv")
    clientes, depositos, parametros, vehiculos, estaciones = cargar_datos()

    mapa = fo.Map(location=[4.5709, -74.2973], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(mapa)

    for idx, fila in verificacion.iterrows():
        secuencia = fila['RouteSequence'].split("-")
        color = colors[idx % len(colors)]
        ruta_coords = []
        for nodo in secuencia:
            if "CD" in nodo:
                info = depositos.loc[depositos['StandardizedID'] == nodo].iloc[0]
            elif "E" in nodo:
                info = estaciones.loc[estaciones['StandardizedID'] == nodo].iloc[0]
            else:
                info = clientes.loc[clientes['StandardizedID'] == nodo].iloc[0]
            coord = [info['Latitude'], info['Longitude']]
            ruta_coords.append(coord)
            fo.Marker(location=coord,
                      popup=f"{nodo} - Vehículo {fila['VehicleId']}",
                      icon=fo.Icon(color=color)).add_to(marker_cluster)
        fo.PolyLine(ruta_coords, color=color, weight=3, opacity=0.8).add_to(mapa)

    for _, est in estaciones.iterrows():
        fo.CircleMarker(
            location=[est['Latitude'], est['Longitude']],
            radius=6,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"Estación {est['StandardizedID']} - Precio: {est['FuelCost']} COP/gal"
        ).add_to(mapa)

    mapa.save(f"{OUTPUT_DIR}/mapa_rutas_caso2.html")
    print("Mapa de rutas guardado en visualizaciones_caso2/")

# ----------------------------------------------------
# 2. Heatmap de precios de combustible
# ----------------------------------------------------
def heatmap_combustible():
    _, _, _, _, estaciones = cargar_datos()
    heat_data = [[row['Latitude'], row['Longitude'], row['FuelCost']] for _, row in estaciones.iterrows()]
    mapa = fo.Map(location=[4.5709, -74.2973], zoom_start=6)
    HeatMap(heat_data, radius=20, blur=15, max_zoom=10).add_to(mapa)
    mapa.save(f"{OUTPUT_DIR}/heatmap_precios_combustible.html")
    print("Heatmap de precios guardado en visualizaciones_caso2/")

# ----------------------------------------------------
# 3. Diagrama de reabastecimiento
# ----------------------------------------------------
def diagrama_reabastecimiento():
    verificacion = pd.read_csv("caso_2/verificacion_caso2.csv")
    plt.figure(figsize=(10, 6))
    for idx, fila in verificacion.iterrows():
        if fila["RefuelStops"] > 0:
            refuels = [float(x) for x in str(fila["RefuelAmounts"]).split("-") if x != "0"]
            plt.plot(range(1, len(refuels) + 1), refuels, marker='o',
                     label=f"{fila['VehicleId']} ({fila['RefuelStops']} paradas)")
    plt.title("Reabastecimientos por Vehículo")
    plt.xlabel("Parada de reabastecimiento")
    plt.ylabel("Combustible cargado (galones)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/diagrama_reabastecimiento.png", dpi=300)
    plt.close()
    print("Diagrama de reabastecimiento guardado en visualizaciones_caso2/")

# ----------------------------------------------------
# 4. Distribución de carga entregada por vehículo
# ----------------------------------------------------
def distribucion_carga():
    datos = pd.read_csv("caso_2/verificacion_caso2.csv")
    datos['DemandSatisfied'] = datos['DemandSatisfied'].fillna('0-0')
    carga_total = datos['DemandSatisfied'].str.split("-").apply(lambda x: sum(map(float, x)))
    plt.figure(figsize=(10, 6))
    plt.pie(carga_total, labels=datos['VehicleId'], autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Carga Entregada por Vehículo')
    plt.savefig(f"{OUTPUT_DIR}/distribucion_carga.png", dpi=300)
    plt.close()
    print("Distribución de carga guardada en visualizaciones_caso2/")

# ----------------------------------------------------
# 5. Fuel level evolution chart per vehicle (simulada)
# ----------------------------------------------------
def evolucion_combustible():
    verificacion = pd.read_csv("caso_2/verificacion_caso2.csv")
    plt.figure(figsize=(10, 6))
    for idx, fila in verificacion.iterrows():
        fuel_cap = fila["FuelCap"]
        stops = fila["RefuelStops"]
        levels = [fuel_cap]  # tanque lleno al inicio
        refuels = [float(x) for x in str(fila["RefuelAmounts"]).split("-") if x != "0"]
        for r in refuels:
            last_level = levels[-1] - (fuel_cap * 0.3) + r  # ejemplo simplificado
            levels.append(max(0, min(fuel_cap, last_level)))
        plt.plot(range(len(levels)), levels, marker='o', label=fila['VehicleId'])
    plt.title("Evolución del Nivel de Combustible por Vehículo")
    plt.xlabel("Segmento de ruta / parada")
    plt.ylabel("Nivel de combustible (galones)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/evolucion_combustible.png", dpi=300)
    plt.close()
    print("Evolución del combustible guardada en visualizaciones_caso2/")

# ----------------------------------------------------
# 6. Cost comparison: optimal refueling vs refueling everywhere
# ----------------------------------------------------
def comparacion_costos():
    verificacion = pd.read_csv("caso_2/verificacion_caso2.csv")
    fuel_prices = cargar_datos()[4]['FuelCost']
    avg_price = np.mean(fuel_prices)
    verificacion['Costo_refuel_all'] = verificacion['FuelCost'] * (avg_price / verificacion['FuelCost'].mean()) * 1.2

    plt.figure(figsize=(10, 6))
    plt.bar(verificacion['VehicleId'], verificacion['TotalCost'], color='green', alpha=0.7, label='Óptimo')
    plt.bar(verificacion['VehicleId'], verificacion['Costo_refuel_all'], color='red', alpha=0.5, label='Repostar en todas las estaciones')
    plt.title("Comparación de Costos: Estrategia Óptima vs. Repostaje Completo")
    plt.xlabel("Vehículo")
    plt.ylabel("Costo total (COP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparacion_costos.png", dpi=300)
    plt.close()
    print("Comparación de costos guardada en visualizaciones_caso2/")

# ----------------------------------------------------
# 7. Menú interactivo
# ----------------------------------------------------
if __name__ == "__main__":
    while True:
        print("\nSeleccione una opción:")
        print("1. Mapa nacional con rutas y estaciones")
        print("2. Heatmap de precios de combustible")
        print("3. Diagrama de reabastecimiento")
        print("4. Distribución de carga entregada")
        print("5. Evolución del nivel de combustible")
        print("6. Comparación de costos (óptimo vs todos)")
        print("7. Salir")

        opcion = input("Ingrese el número de la opción deseada: ")

        if opcion == '1':
            mapa_rutas()
        elif opcion == '2':
            heatmap_combustible()
        elif opcion == '3':
            diagrama_reabastecimiento()
        elif opcion == '4':
            distribucion_carga()
        elif opcion == '5':
            evolucion_combustible()
        elif opcion == '6':
            comparacion_costos()
        elif opcion == '7':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")
