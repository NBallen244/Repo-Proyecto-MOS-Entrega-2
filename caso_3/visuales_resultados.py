import os
import sys
import pandas as pd
import folium as fo
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from carga_datos.cargaDatos import cargar_datos_caso3 as cargar_datos

OUTPUT_DIR = "caso_3/visualizaciones_caso3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

colors = ['blue','green','purple','orange','darkred','cadetblue','black','gray']

# =====================================================
# TOLL IMPACT ANALYSIS
# =====================================================

def toll_cost_breakdown():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    plt.figure(figsize=(10,6))

    tolls = df["TollCost"]
    total = df["TotalCost"]
    fuel = df["FuelCost"]
    others = total - tolls - fuel

    x = np.arange(len(df))
    width = 0.35

    plt.bar(x, tolls, width, label="Peajes")
    plt.bar(x, fuel, width, bottom=tolls, label="Combustible")
    plt.bar(x, others, width, bottom=tolls+fuel, label="Otros")

    plt.xticks(x, df["VehicleId"])
    plt.ylabel("Costo (COP)")
    plt.title("Impacto de Peajes vs Otros Costos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/toll_cost_impact.png", dpi=300)
    plt.close()


def toll_avoidance_analysis():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")

    df["AvoidanceIndicator"] = df.apply(lambda r: 1 if r["TollsVisited"] == 0 else 0, axis=1)

    plt.figure(figsize=(10,6))
    plt.bar(df["VehicleId"], df["AvoidanceIndicator"], color="red")
    plt.title("Indicador de Estrategias de Evitación de Peajes (1 = evitó peajes)")
    plt.savefig(f"{OUTPUT_DIR}/toll_avoidance_indicator.png", dpi=300)
    plt.close()


def weight_based_toll_optimization():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    toll_weights = df["VehicleWeights"].str.split("-").apply(lambda x: max([float(i) for i in x]) if x != ['0'] else 0)
    plt.figure(figsize=(10,6))
    plt.scatter(toll_weights, df["TollCost"], s=80)
    plt.title("Relación Peso vs Costo de Peajes")
    plt.xlabel("Peso vehicular al cruzar peajes (kg)")
    plt.ylabel("Costo de peajes (COP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/toll_vs_weight.png", dpi=300)
    plt.close()


# =====================================================
# WEIGHT COMPLIANCE REPORT
# =====================================================

def weight_restriction_map():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    clientes, depositos, parametros, vehiculos, estaciones, peajes = cargar_datos()

    mapa = fo.Map(location=[4.5709,-74.2973], zoom_start=6)

    restricted = clientes[~clientes["MaxWeight"].isna()]
    compliant = clientes[clientes["MaxWeight"].isna()]

    for _, row in restricted.iterrows():
        fo.Marker(
            [row["Latitude"],row["Longitude"]],
            popup=f"{row['StandardizedID']} - MaxWeight: {row['MaxWeight']}",
            icon=fo.Icon(color="red")
        ).add_to(mapa)

    mapa.save(f"{OUTPUT_DIR}/weight_restriction_map.html")


def impact_on_route_structure():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    plt.figure(figsize=(10,6))
    plt.bar(df["VehicleId"], df["Municipalities"], color="blue")
    plt.title("Impacto de Restricciones: Cantidad de Municipios Visitados vs Peso")
    plt.savefig(f"{OUTPUT_DIR}/restriction_impact.png", dpi=300)
    plt.close()


# =====================================================
# SENSITIVITY ANALYSIS
# =====================================================

def sensitivity_fuel_variation():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    df["FuelPlus20"] = df["FuelCost"] * 1.2
    df["FuelMinus20"] = df["FuelCost"] * 0.8

    plt.figure(figsize=(10,6))
    plt.plot(df["VehicleId"], df["FuelCost"], marker="o", label="Actual")
    plt.plot(df["VehicleId"], df["FuelPlus20"], marker="o", label="+20%")
    plt.plot(df["VehicleId"], df["FuelMinus20"], marker="o", label="-20%")
    plt.title("Análisis de Sensibilidad: Precio del Combustible ±20%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sensitivity_fuel.png", dpi=300)
    plt.close()


def sensitivity_remove_tolls():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    df["WithoutTolls"] = df["TotalCost"] - df["TollCost"]

    plt.figure(figsize=(10,6))
    plt.plot(df["VehicleId"], df["TotalCost"], marker="o", label="Con Peajes")
    plt.plot(df["VehicleId"], df["WithoutTolls"], marker="o", label="Sin Peajes")
    plt.title("Impacto de Eliminar Peajes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sensitivity_remove_tolls.png", dpi=300)
    plt.close()


def sensitivity_tighten_restrictions():
    df = pd.read_csv("caso_3/verificacion_caso3.csv")
    df["Restricted"] = df["Municipalities"] * 0.8

    plt.figure(figsize=(10,6))
    plt.plot(df["VehicleId"], df["Municipalities"], marker="o", label="Actual")
    plt.plot(df["VehicleId"], df["Restricted"], marker="o", label="Límites Más Fuertes")
    plt.title("Impacto de Restricciones Más Severas")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/sensitivity_tighten_restrictions.png", dpi=300)
    plt.close()

def mapa_rutas():
    verificacion = pd.read_csv("caso_3/verificacion_caso3.csv")
    clientes, depositos, parametros, vehiculos, estaciones, peajes = cargar_datos()

    mapa = fo.Map(location=[4.5709, -74.2973], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(mapa)

    for idx, fila in verificacion.iterrows():
        secuencia = fila['RouteSeq'].split("-")
        color = colors[idx % len(colors)]
        ruta_coords = []

        for nodo in secuencia:
            if nodo.startswith("CD"):
                info = depositos[depositos["StandardizedID"] == nodo].iloc[0]
            elif nodo.startswith("E"):
                info = estaciones[estaciones["StandardizedID"] == nodo].iloc[0]
            elif nodo.startswith("P"):
                info = peajes[peajes["StandardizedID"] == nodo].iloc[0]
            else:
                info = clientes[clientes["StandardizedID"] == nodo].iloc[0]

            coord = [info["Latitude"], info["Longitude"]]
            ruta_coords.append(coord)
            fo.Marker(location=coord,
                      popup=f"{nodo} - Vehículo {fila['VehicleId']}",
                      icon=fo.Icon(color=color)).add_to(marker_cluster)

        fo.PolyLine(ruta_coords, color=color, weight=3, opacity=0.8).add_to(mapa)

    # Puntos especiales de estaciones
    for _, est in estaciones.iterrows():
        fo.CircleMarker(
            location=[est['Latitude'], est['Longitude']],
            radius=6,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"Estación {est['StandardizedID']} - Precio: {est['FuelCost']} COP/gal"
        ).add_to(mapa)

    # Iconos de peajes
    for _, p in peajes.iterrows():
        fo.Marker(
            location=[p['Latitude'], p['Longitude']],
            popup=f"Peaje {p['StandardizedID']} - Base: {p['BaseRate']} + {p['RatePerTon']} COP/t",
            icon=fo.Icon(color='darkred', icon='road', prefix='fa')
        ).add_to(mapa)

    mapa.save(f"{OUTPUT_DIR}/mapa_rutas_caso3.html")
    print("Mapa de rutas guardado en visualizaciones_caso3/")

# =====================================================
# MENÚ
# =====================================================

if __name__ == "__main__":
    functions = [
        toll_cost_breakdown,
        toll_avoidance_analysis,
        weight_based_toll_optimization,
        weight_restriction_map,
        impact_on_route_structure,
        sensitivity_fuel_variation,
        sensitivity_remove_tolls,
        sensitivity_tighten_restrictions,
        mapa_rutas
    ]

    print("\nGenerando visualizaciones del Caso 3...")
    for fn in functions:
        fn()
    print("Visualizaciones completadas en caso_3/visualizaciones_caso3/")
