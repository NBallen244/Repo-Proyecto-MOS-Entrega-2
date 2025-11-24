import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import os, sys
import time
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)
from carga_datos.cargaDatos import cargar_datos_caso3




def construccion_modelo(clientes, depositos, parametros, vehiculos, estaciones, peajes, matriz_df):
    # -----------------------------
    # 1. PARÁMETROS Y CONJUNTOS
    # -----------------------------
    clientes_ids = clientes['StandardizedID'].tolist()
    vehiculos_ids = vehiculos['StandardizedID'].tolist()
    estaciones_ids = estaciones['StandardizedID'].tolist()
    peajes_ids = peajes['StandardizedID'].tolist()
    depot_id = depositos['StandardizedID'].iloc[0]

    N = [depot_id] + clientes_ids + estaciones_ids + peajes_ids

    # Parámetros de costos
    C_fixed = float(parametros.loc[parametros['Parameter'] == 'C_fixed', 'Value'].values[0])
    C_dist = float(parametros.loc[parametros['Parameter'] == 'C_dist', 'Value'].values[0])
    C_time = float(parametros.loc[parametros['Parameter'] == 'C_time', 'Value'].values[0])
    efficiency = 9.0  # km/gal promedio

    # Costos y restricciones
    fuel_price_station = dict(zip(estaciones['StandardizedID'], estaciones['FuelCost']))
    base_rate_toll = dict(zip(peajes['StandardizedID'], peajes['BaseRate']))
    rate_per_ton_toll = dict(zip(peajes['StandardizedID'], peajes['RatePerTon']))
    max_weight_city = dict(zip(clientes['StandardizedID'], clientes['MaxWeight']))

    demand = dict(zip(clientes['StandardizedID'], clientes['Demand']))
    capacity_v = dict(zip(vehiculos['StandardizedID'], vehiculos['Capacity']))
    range_v = dict(zip(vehiculos['StandardizedID'], vehiculos['Range']))

    # Distancias y tiempos
    dist = {(row['FromID'], row['ToID']): row['Distance_km'] for _, row in matriz_df.iterrows()}
    time = {(row['FromID'], row['ToID']): row['Time_min'] for _, row in matriz_df.iterrows()}

    # -----------------------------
    # 2. MODELO PYOMO
    # -----------------------------
    model = ConcreteModel()

    model.N = Set(initialize=N)
    model.U = Set(initialize=clientes_ids)
    model.E = Set(initialize=estaciones_ids)
    model.P = Set(initialize=peajes_ids)
    model.V = Set(initialize=vehiculos_ids)
    model.CD = Set(initialize=[depot_id])

    model.d = Param(model.N, model.N, initialize=dist, default=0)
    model.t = Param(model.N, model.N, initialize=time, default=0)
    model.demanda = Param(model.U, initialize=demand, default=0)
    model.capacidad = Param(model.V, initialize=capacity_v)
    model.rango = Param(model.V, initialize=range_v)

    # -----------------------------
    # 3. VARIABLES
    # -----------------------------
    model.X = Var(model.N, model.N, model.V, within=Binary)
    model.Y = Var(model.N, model.V, within=Binary)
    model.A = Var(model.V, within=Binary)
    model.F = Var(model.N, model.V, within=NonNegativeReals)
    model.R = Var(model.N, model.V, within=NonNegativeReals)
    model.Carga = Var(model.N, model.V, within=NonNegativeReals)
    model.Z = Var(model.P, model.V, within=NonNegativeReals)
    model.Uaux = Var(model.N, model.V, within=NonNegativeReals)

    # -----------------------------
    # 4. FUNCIÓN OBJETIVO
    # -----------------------------
    def obj_rule(model):
        cost_fixed = sum(C_fixed * model.A[v] for v in model.V)
        cost_dist = sum(C_dist * model.d[i, j] * model.X[i, j, v]
                        for v in model.V for i in model.N for j in model.N if i != j)
        cost_time = sum(C_time * (model.t[i, j] / 60) * model.X[i, j, v]
                        for v in model.V for i in model.N for j in model.N if i != j)
        cost_fuel = sum(model.R[i, v] * fuel_price_station.get(i, 0)
                        for v in model.V for i in model.E)
        cost_toll = sum(
            base_rate_toll.get(p, 0) * model.Y[p, v] +
            (rate_per_ton_toll.get(p, 0) / 1000) * model.Z[p, v]
            for v in model.V for p in model.P
        )
        return cost_fixed + cost_dist + cost_time + cost_fuel + cost_toll

    model.obj = Objective(rule=obj_rule, sense=minimize)

        # -----------------------------
    # 5. RESTRICCIONES
    # -----------------------------
    M = 1e5
    depot = list(model.CD)[0]

    # Activación del vehículo
    def activation_rule(model, v):
        return model.A[v] >= sum(model.Y[i, v] for i in model.N) / len(model.N)
    model.activation = Constraint(model.V, rule=activation_rule)

    # Salida del depósito una sola vez
    def start_rule(model, v):
        return sum(model.X[depot, j, v] for j in model.N if j != depot) == model.A[v]
    model.start = Constraint(model.V, rule=start_rule)

    # Entrada al depósito una sola vez
    def end_rule(model, v):
        return sum(model.X[i, depot, v] for i in model.N if i != depot) == model.A[v]
    model.end = Constraint(model.V, rule=end_rule)

    # Prohibir visitas intermedias al depósito (solo inicio y fin)
    def depot_once_rule(model, v):
        return sum(model.X[i, depot, v] + model.X[depot, i, v] for i in model.N if i != depot) <= 2
    model.depot_once = Constraint(model.V, rule=depot_once_rule)

    # Cobertura de clientes
    def cover_rule(model, i):
        return sum(model.X[j, i, v] for v in model.V for j in model.N if j != i) == 1
    model.cover = Constraint(model.U, rule=cover_rule)

    # Flujo de continuidad
    def flow_rule(model, j, v):
        return sum(model.X[i, j, v] for i in model.N if i != j) == sum(model.X[j, k, v] for k in model.N if k != j)
    model.flow = Constraint(model.N, model.V, rule=flow_rule)

    # Capacidad máxima del vehículo
    def capacity_rule(model, v):
        return sum(model.demanda[i] * sum(model.X[i, j, v] for j in model.N if j != i)
                   for i in model.U) <= model.capacidad[v]
    model.cap = Constraint(model.V, rule=capacity_rule)

    # ---- CARGA INICIAL: cada vehículo arranca con su capacidad máxima ----
    def init_weight_rule(model, v):
        return model.Carga[depot, v] == model.capacidad[v]
    model.init_weight = Constraint(model.V, rule=init_weight_rule)

    # Balance dinámico de carga
    def weight_balance_rule(model, i, j, v):
        if i != j and j in model.U:
            return model.Carga[j, v] >= model.Carga[i, v] - model.demanda[j] * model.X[i, j, v] - (1 - model.X[i, j, v]) * M
        return Constraint.Skip
    model.weight_balance = Constraint(model.N, model.N, model.V, rule=weight_balance_rule)

    # Restricción de peso municipal
    def max_weight_rule(model, i, v):
        if i in max_weight_city and not pd.isna(max_weight_city[i]):
            return model.Carga[i, v] <= max_weight_city[i]
        return Constraint.Skip
    model.max_weight = Constraint(model.U, model.V, rule=max_weight_rule)

    # Combustible
    def fuel_balance_rule(model, i, j, v):
        if i != j:
            return model.F[j, v] >= model.F[i, v] - (model.d[i, j] / efficiency) + model.R[j, v] - (1 - model.X[i, j, v]) * M
        return Constraint.Skip
    model.fuel_balance = Constraint(model.N, model.N, model.V, rule=fuel_balance_rule)

    def fuel_cap_rule(model, i, v):
        return model.F[i, v] <= model.rango[v] / efficiency
    model.fuel_cap = Constraint(model.N, model.V, rule=fuel_cap_rule)

    def init_fuel_rule(model, v):
        return model.F[depot, v] == model.rango[v] / efficiency
    model.init_fuel = Constraint(model.V, rule=init_fuel_rule)

    # Repostaje solo en estaciones
    def refuel_stations_rule(model, i, v):
        if i not in estaciones_ids:
            return model.R[i, v] == 0
        return Constraint.Skip
    model.refuel_stations = Constraint(model.N, model.V, rule=refuel_stations_rule)

    # Peajes linealizados
    def z_link1_rule(model, p, v):
        return model.Z[p, v] <= model.Carga[p, v]
    def z_link2_rule(model, p, v):
        return model.Z[p, v] <= M * model.Y[p, v]
    def z_link3_rule(model, p, v):
        return model.Z[p, v] >= model.Carga[p, v] - M * (1 - model.Y[p, v])
    model.z_link1 = Constraint(model.P, model.V, rule=z_link1_rule)
    model.z_link2 = Constraint(model.P, model.V, rule=z_link2_rule)
    model.z_link3 = Constraint(model.P, model.V, rule=z_link3_rule)

    # MTZ sin depot
    def mtz_rule(model, i, j, v):
        if i != j and i not in model.CD and j not in model.CD:
            return model.Uaux[i, v] - model.Uaux[j, v] + len(model.N) * model.X[i, j, v] <= len(model.N) - 1
        return Constraint.Skip
    model.mtz = Constraint(model.N, model.N, model.V, rule=mtz_rule)


    # Devolver modelo y parámetros clave
    return model, C_fixed, C_dist, C_time, efficiency, fuel_price_station, base_rate_toll, rate_per_ton_toll


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    clientes, depositos, parametros, vehiculos, estaciones, peajes = cargar_datos_caso3()
    matriz_df = pd.read_csv("caso_3/matriz.csv")

    (
        model,
        C_fixed,
        C_dist,
        C_time,
        efficiency,
        fuel_price_station,
        base_rate_toll,
        rate_per_ton_toll
    ) = construccion_modelo(clientes, depositos, parametros, vehiculos, estaciones, peajes, matriz_df)

    solver = SolverFactory("appsi_highs")
    results = solver.solve(model, tee=True, timelimit=15)

    print("\n*** RESULTADOS: LOGISTICO CASO 3 (DINÁMICA DE PESO, PEAJES Y REABASTECIMIENTO) ***")
    print(f"Costo total: {value(model.obj):,.2f} COP\n")

    os.makedirs("caso_3", exist_ok=True)
    output_path = "caso_3/verificacion_caso3.csv"
    registros = []

    for v in model.V:
        rutas = [(i, j) for i in model.N for j in model.N if i != j and value(model.X[i, j, v]) > 0.5]
        if not rutas:
            continue

        route_nodes = [rutas[0][0]] + [j for (_, j) in rutas]
        route_seq = "-".join(route_nodes)
        municipios = [n for n in route_nodes if n in clientes['StandardizedID'].values]
        demandas = [str(int(model.demanda[m])) if m in model.U else "0" for m in municipios]
        tolls = [n for n in route_nodes if n in peajes['StandardizedID'].values]
        estaciones_v = [n for n in route_nodes if n in estaciones['StandardizedID'].values]

        init_load = sum(model.demanda[i] for i in model.U)
        init_fuel = model.rango[v] / efficiency
        refuel_stops = len([i for i in estaciones_v if value(model.R[i, v]) > 0])
        refuel_amounts = "-".join([str(round(value(model.R[i, v]), 2)) for i in estaciones_v if value(model.R[i, v]) > 0]) or "0"
        toll_costs = "-".join([str(round(base_rate_toll.get(p, 0) + (rate_per_ton_toll.get(p, 0) / 1000) * value(model.Z[p, v]), 2)) for p in tolls]) or "0"
        weights = "-".join([str(round(value(model.Carga[i, v]), 2)) for i in municipios]) or "0"

        dist_total = sum(value(model.d[i, j]) for (i, j) in rutas)
        time_total = sum(value(model.t[i, j]) for (i, j) in rutas)
        fuel_cost = sum(value(model.R[i, v]) * fuel_price_station.get(i, 0) for i in model.E)
        toll_cost_total = sum(base_rate_toll.get(p, 0) + (rate_per_ton_toll.get(p, 0) / 1000) * value(model.Z[p, v]) for p in tolls)
        total_cost = C_fixed + C_dist * dist_total + C_time * (time_total / 60) + fuel_cost + toll_cost_total

        registros.append({
            "VehicleId": v,
            "LoadCap": value(model.capacidad[v]),
            "FuelCap": value(model.rango[v]) / efficiency,
            "RouteSeq": route_seq,
            "Municipalities": len(municipios),
            "Demand": "-".join(demandas),
            "InitLoad": value(init_load),
            "InitFuel": value(init_fuel),
            "RefuelStops": refuel_stops,
            "RefuelAmounts": refuel_amounts,
            "TollsVisited": len(tolls),
            "TollCosts": toll_costs,
            "VehicleWeights": weights,
            "Distance": round(dist_total, 2),
            "Time": round(time_total, 2),
            "FuelCost": round(fuel_cost, 2),
            "TollCost": round(toll_cost_total, 2),
            "TotalCost": round(total_cost, 2)
        })

    df_result = pd.DataFrame(registros)
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Archivo de verificación generado en: {output_path}")