import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import os, sys
import time
# ----------------------------------------------------
# 1. IMPORTAR CARGA DE DATOS
# ----------------------------------------------------
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from carga_datos.cargaDatos import cargar_datos_caso2 as cargar_datos


# =====================================================
# FUNCIÓN PRINCIPAL DE CONSTRUCCIÓN DEL MODELO
# =====================================================
def construccion_modelo(matriz_path="caso_2/matriz.csv"):
    clientes, depositos, parametros, vehiculos, estaciones = cargar_datos()
    matriz = pd.read_csv(matriz_path)
    matriz['FromID'] = matriz['FromID'].astype(str)
    matriz['ToID'] = matriz['ToID'].astype(str)

    clientes_ids = clientes['StandardizedID'].tolist()
    vehiculos_ids = vehiculos['StandardizedID'].tolist()
    estaciones_ids = estaciones['StandardizedID'].tolist()
    depot_id = depositos['StandardizedID'].iloc[0]
    N = [depot_id] + clientes_ids + estaciones_ids

    C_fixed = float(parametros.loc[parametros['Parameter'] == 'C_fixed', 'Value'].values[0])
    C_dist = float(parametros.loc[parametros['Parameter'] == 'C_dist', 'Value'].values[0])
    C_time = float(parametros.loc[parametros['Parameter'] == 'C_time', 'Value'].values[0])
    efficiency = 9.0

    fuel_price_station = dict(zip(estaciones['StandardizedID'], estaciones['FuelCost']))
    demand = dict(zip(clientes['StandardizedID'], clientes['Demand']))
    capacity_v = dict(zip(vehiculos['StandardizedID'], vehiculos['Capacity']))
    range_v = dict(zip(vehiculos['StandardizedID'], vehiculos['Range']))

    dist, time = {}, {}
    for _, row in matriz.iterrows():
        i, j = row['FromID'], row['ToID']
        if i in N and j in N:
            dist[(i, j)] = float(row['Distance_km'])
            time[(i, j)] = float(row['Time_min'])
    for n in N:
        dist[(n, n)] = 0
        time[(n, n)] = 0

    model = ConcreteModel()
    model.N = Set(initialize=N)
    model.U = Set(initialize=clientes_ids)
    model.E = Set(initialize=estaciones_ids)
    model.V = Set(initialize=vehiculos_ids)
    model.CD = Set(initialize=[depot_id])

    model.d = Param(model.N, model.N, initialize=dist)
    model.t = Param(model.N, model.N, initialize=time)
    model.demanda = Param(model.U, initialize=demand)
    model.capacidad = Param(model.V, initialize=capacity_v)
    model.rango = Param(model.V, initialize=range_v)

    model.X = Var(model.N, model.N, model.V, within=Binary)
    model.Y = Var(model.N, model.V, within=Binary)
    model.F = Var(model.N, model.V, within=NonNegativeReals)
    model.R = Var(model.N, model.V, within=NonNegativeReals)
    model.Carga = Var(model.N, model.V, within=NonNegativeReals)
    model.Uaux = Var(model.N, model.V, within=NonNegativeReals)

    def obj_rule(model):
        cost_fixed = sum(C_fixed * model.Y[depot_id, v] for v in model.V)
        cost_dist = sum(C_dist * model.d[i, j] * model.X[i, j, v]
                        for v in model.V for i in model.N for j in model.N if i != j)
        cost_time = sum(C_time * (model.t[i, j] / 60) * model.X[i, j, v]
                        for v in model.V for i in model.N for j in model.N if i != j)
        cost_fuel = sum(model.R[i, v] * fuel_price_station.get(i, 0)
                        for v in model.V for i in model.E)
        return cost_fixed + cost_dist + cost_time + cost_fuel

    model.obj = Objective(rule=obj_rule, sense=minimize)

    def start_rule(model, v):
        return sum(model.X[depot_id, j, v] for j in model.N if j != depot_id) == 1
    model.start = Constraint(model.V, rule=start_rule)

    def end_rule(model, v):
        return sum(model.X[i, depot_id, v] for i in model.N if i != depot_id) == 1
    model.end = Constraint(model.V, rule=end_rule)

    def cover_rule(model, i):
        return sum(model.X[j, i, v] for v in model.V for j in model.N if j != i) == 1
    model.cover = Constraint(model.U, rule=cover_rule)

    def flow_rule(model, j, v):
        return sum(model.X[i, j, v] for i in model.N if i != j) == sum(model.X[j, k, v] for k in model.N if k != j)
    model.flow = Constraint(model.N, model.V, rule=flow_rule)

    def capacity_rule(model, v):
        return sum(model.demanda[i] * sum(model.X[i, j, v] for j in model.N if j != i)
                   for i in model.U) <= model.capacidad[v]
    model.cap = Constraint(model.V, rule=capacity_rule)

    def fuel_balance_rule(model, i, j, v):
        if i != j:
            return model.F[j, v] >= model.F[i, v] - (model.d[i, j] / efficiency) + model.R[j, v] - (1 - model.X[i, j, v]) * 1e4
        return Constraint.Skip
    model.fuel_balance = Constraint(model.N, model.N, model.V, rule=fuel_balance_rule)

    def fuel_cap_rule(model, i, v):
        return model.F[i, v] <= model.rango[v] / efficiency
    model.fuel_cap = Constraint(model.N, model.V, rule=fuel_cap_rule)

    def init_fuel_rule(model, v):
        return model.F[depot_id, v] == model.rango[v] / efficiency
    model.init_fuel = Constraint(model.V, rule=init_fuel_rule)

    def refuel_stations_rule(model, i, v):
        if i not in estaciones_ids:
            return model.R[i, v] == 0
        return Constraint.Skip
    model.refuel_stations = Constraint(model.N, model.V, rule=refuel_stations_rule)

    def mtz_rule(model, i, j, v):
        if i != j and i not in model.CD and j not in model.CD:
            return model.Uaux[i, v] - model.Uaux[j, v] + len(model.N) * model.X[i, j, v] <= len(model.N) - 1
        return Constraint.Skip
    model.mtz = Constraint(model.N, model.N, model.V, rule=mtz_rule)

    return model, depot_id, efficiency, fuel_price_station, clientes, vehiculos, C_fixed, C_dist, C_time


# =====================================================
# EJECUCIÓN Y ARCHIVO DE VERIFICACIÓN
# =====================================================
if __name__ == "__main__":
    model, depot_id, efficiency, fuel_price_station, clientes, vehiculos, C_fixed, C_dist, C_time = construccion_modelo("caso_2/matriz.csv")

    solver = SolverFactory("appsi_highs")
    start_time = time.time()
    results = solver.solve(model, tee=True, timelimit=600)
    end_time = time.time()

    print("\n*** RESULTADOS: LOGISTICO CASO 2 (REABASTECIMIENTO ESTRATÉGICO) ***")
    total_cost = value(model.obj)
    print(f"Coste total: {total_cost:,.2f} COP\n")

    registros = []
    for v in model.V:
        rutas = [(i, j) for i in model.N for j in model.N if i != j and value(model.X[i, j, v]) > 0.5]
        if not rutas:
            continue

        route_seq = [depot_id]
        current = depot_id
        while True:
            next_nodes = [j for (i, j) in rutas if i == current]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            route_seq.append(next_node)
            current = next_node
            if next_node == depot_id:
                break
        route_str = "-".join(route_seq)

        clients_in_route = [n for n in route_seq if n in model.U]
        demands = [str(int(value(model.demanda[i]))) if i in model.U else "0" for i in route_seq]
        refuels = [(i, round(value(model.R[i, v]), 2)) for i in model.E if value(model.R[i, v]) > 0]
        total_distance = sum(value(model.d[i, j]) for (i, j) in rutas)
        total_time = sum(value(model.t[i, j]) for (i, j) in rutas)
        fuel_cost = sum(r[1] * fuel_price_station[r[0]] for r in refuels)

        # Cálculo de costo individual
        cost_fixed = C_fixed
        cost_dist = C_dist * total_distance
        cost_time = C_time * (total_time / 60)
        total_cost_v = cost_fixed + cost_dist + cost_time + fuel_cost

        registros.append({
            "VehicleId": v,
            "LoadCap": value(model.capacidad[v]),
            "FuelCap": round(value(model.rango[v]) / efficiency, 2),
            "RouteSequence": route_str,
            "Municipalities": len(clients_in_route),
            "DemandSatisfied": "-".join(demands),
            "InitLoad": value(model.capacidad[v]),
            "InitFuel": round(value(model.rango[v]) / efficiency, 2),
            "RefuelStops": len(refuels),
            "RefuelAmounts": "-".join([str(r[1]) for r in refuels]) if refuels else "0",
            "Distance": round(total_distance, 2),
            "Time": round(total_time, 2),
            "FuelCost": round(fuel_cost, 2),
            "TotalCost": round(total_cost_v, 2)
        })

    df = pd.DataFrame(registros)
    df.to_csv("caso_2/verificacion_caso2.csv", index=False)
    print("\nArchivo 'verificacion_caso2.csv' creado correctamente")
