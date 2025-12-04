from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd

import sys
import os
import time

# Add the directory containing cargaBase.py to Python's search path
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from carga_datos.cargaDatos import cargar_datos_base as cargar_datos

def construccion_modelo(clientes, depositos, parametros, vehiculos):
    # Aquí iría la construcción del modelo de optimización usando Pyomo
    model = ConcreteModel()
    
    # Definición de conjuntos
    #Clientes
    model.C = Set(initialize=clientes['StandardizedID'].tolist())
    #Depósitos
    model.D = Set(initialize=depositos['StandardizedID'].tolist())
    #Vehículos
    model.V = Set(initialize=vehiculos['StandardizedID'].tolist())
    # Nodos
    model.N = model.C | model.D
    
    # Definición de parámetros
    #distancias y tiempos entre nodos
    distancia_dict = {}
    #tiempos entre nodos
    tiempo_dict = {}
    #primero de los centros de distribucion a todos los clientes
    matriz_distancia_tiempo = pd.read_csv("caso_base/matriz.csv")
    for i in model.D:
        for j in model.C:
            distancia = matriz_distancia_tiempo.loc[
                (matriz_distancia_tiempo['FromID'] == i) & (matriz_distancia_tiempo['ToID'] == j),
                'Distance_km'
            ].values[0]
            duracion = matriz_distancia_tiempo.loc[
                (matriz_distancia_tiempo['FromID'] == i) & (matriz_distancia_tiempo['ToID'] == j),
                'Time_min'
            ].values[0]
            distancia_dict[(i,j)] = distancia
            tiempo_dict[(i,j)] = duracion
    #luego entre clientes
    for i in model.C:
        for j in model.C:
            if i != j:
                distancia = matriz_distancia_tiempo.loc[
                    (matriz_distancia_tiempo['FromID'] == i) & (matriz_distancia_tiempo['ToID'] == j),
                    'Distance_km'
                ].values[0]
                duracion = matriz_distancia_tiempo.loc[
                    (matriz_distancia_tiempo['FromID'] == i) & (matriz_distancia_tiempo['ToID'] == j),
                    'Time_min'
                ].values[0]
                distancia_dict[(i,j)] = distancia
                tiempo_dict[(i,j)] = duracion
    model.dist=Param(model.N, model.N, initialize=distancia_dict, default=0)
    model.tiempo=Param(model.N, model.N, initialize=tiempo_dict, default=0)
    #Demandas de clientes (kg)
    model.demand=Param(model.C, initialize=clientes.set_index('StandardizedID')['Demand'].to_dict())
    #Capacidad de los vehículos (kg)
    model.cap=Param(model.V, initialize=vehiculos.set_index('StandardizedID')['Capacity'].to_dict())
    #Autonomia de los vehículos (km por tanque lleno)
    model.aut=Param(model.V, initialize=vehiculos.set_index('StandardizedID')['Range'].to_dict())
    #Costo por galon de gasolina (COP/galon)
    model.gas_cost=Param(initialize=parametros.loc[parametros['Parameter'] == 'fuel_price', 'Value'].values[0])
    #Rendimiento gasolina (km/galon)
    model.gas_eff=Param(initialize=parametros.loc[parametros['Parameter'] == 'fuel_efficiency_typical', 'Value'].values[0])
    
    #Variables de decisión
    #evitar variables innecesarias (i != j)
    valid_arcs = [(i, j, v) for i in model.N for j in model.N for v in model.V if i != j]
    model.x = Var(valid_arcs, domain=Binary)  # 1 si el vehículo v va de i a j
    model.y = Var(model.N, model.V, domain=NonNegativeIntegers)  # entero que representa el orden en el que el vehiculo visita el nodo
    #funcion objetivo
    def obj_rule(model):
        return sum(model.dist[i,j] * (model.gas_cost / model.gas_eff) * model.x[i,j,v] 
                   for i in model.N for j in model.N for v in model.V if i != j)
    model.OBJ = Objective(rule=obj_rule, sense=minimize)
    
    #Restricciones
    
    #Cada cliente es atendido por exactamente un vehículo
    def cliente_atendido_rule(model, c):
        return sum(model.x[i,c,v] for i in model.N for v in model.V if i != c) == 1
    model.cliente_atendido = Constraint(model.C, rule=cliente_atendido_rule)
    #Un vehiculo no puede recorrer mas distancia que su autonomia
    def autonomia_vehiculo_rule(model, v):
        return sum(model.dist[i,j] * model.x[i,j,v] for i in model.N for j in model.N if i != j) <= model.aut[v]
    model.autonomia_vehiculo = Constraint(model.V, rule=autonomia_vehiculo_rule)
    #todo vehiculo debe partir de un deposito
    def salida_deposito_rule(model, v):
        return sum(model.x[d,j,v] for d in model.D for j in model.N if j != d) == 1
    model.salida_deposito = Constraint(model.V, rule=salida_deposito_rule)
    #todo vehiculo debe regresar a un deposito
    def regreso_deposito_rule(model, v):
        return sum(model.x[i,d,v] for d in model.D for i in model.N if i != d) == 1
    model.regreso_deposito = Constraint(model.V, rule=regreso_deposito_rule)
    #Todo vehiculo que llege a un nodo que no sea deposito debe salir de ese nodo
    def flujo_nodos_rule(model, n, v):
        if n in model.D:
            return Constraint.Skip
        return sum(model.x[i,n,v] for i in model.N if i != n) == sum(model.x[n,j,v] for j in model.N if j != n)
    model.flujo_nodos = Constraint(model.N, model.V, rule=flujo_nodos_rule)
    #Eliminacion de subciclos (MTZ)
    def mtz_rule(model, i, j, v):
        if i == j or i in model.D or j in model.D:
            return Constraint.Skip
        M = len(model.C) + 1
        return model.y[i,v] - model.y[j,v] + M * model.x[i,j,v] <= M - 1
    model.mtz = Constraint(model.N, model.N, model.V, rule=mtz_rule)
    #los centros de distribucion tienen orden 0 (inicio del recorrido)
    def orden_deposito_rule(model, d, v):
        return model.y[d,v] == 0
    model.orden_deposito = Constraint(model.D, model.V, rule=orden_deposito_rule)
    #Capacidad de los vehiculos no supera la demanda de atendidos
    def capacidad_vehiculo_rule(model, v):
        return sum(model.demand[c] * model.x[i,c,v] for c in model.C for i in model.N if i != c) <= model.cap[v]
    model.capacidad_vehiculo = Constraint(model.V, rule=capacidad_vehiculo_rule)
    #Dominio de variables ya definido en la declaración
    print("Modelo construido exitosamente.")
    return model

if __name__ == "__main__":
    # Cargar los datos
    clientes, depositos, parametros, vehiculos = cargar_datos()
    # Construir el modelo
    model = construccion_modelo(clientes, depositos, parametros, vehiculos)
    # Crear el solucionador
    solver = SolverFactory('')  # Asegúrate de tener SCIP instalado
    #solver.options['time_limit'] = 3600
    
    # Resolver el modelo
    start_time = time.time()
    results = solver.solve(model, tee=True)
    end_time = time.time()
    
    print(f"Tiempo de resolución: {end_time - start_time} segundos")
    
    #diccionario resultados
    resultados = {
        "VehicleId": [],
        "LoadCap": [],
        "FuelCap": [],
        "RouteSequence": [],
        "Municipalities": [],
        "DemandSatisfied": [],
        "InitialLoad": [],
        "InitialFuel": [],
        "Distance": [],
        "Time": [],
        "TotalCost": []
    }
    
    #Contruccion archivo de verificacion de resultados
    for vehicleid in model.V:
        ruta = []
        #solo hay 1 centro de distribucion
        for deposit in model.D:
            ruta.append([deposit, 0])
        municipios = 0
        demandas_satisfecha = []
        demanda_total=0
        
        distancia_recorrida = 0
        tiempo_total=0
        #revisamos todos los clientes
        for i in model.C:
            #si pasa por el cliente (tiene un valor en el orden de su ruta mayor a 0)
            if model.y[i,vehicleid].value is not None and model.y[i,vehicleid].value > 0:
                #se agrega a la ruta, asi como su municipio y demanda
                ruta.append([i, int(model.y[i,vehicleid].value)])
                municipios += 1
                demandas_satisfecha.append((str(value(model.demand[i])), int(model.y[i,vehicleid].value)))
                demanda_total += value(model.demand[i])
        #ordenar en base a y (orden de ruta)     
        ruta = sorted(ruta, key=lambda x: x[1])
        demandas_satisfecha = sorted(demandas_satisfecha, key=lambda x: x[1])
        demandas_satisfecha = [d[0] for d in demandas_satisfecha]
        if len(demandas_satisfecha) == 0:
            demandas_satisfecha.append('0')
        ruta = [node[0] for node in ruta]
        #agregamos el deposito al final de la ruta
        ruta.append(ruta[0])
        #calculamos la distancia de la ruta
        if len(ruta)>2:
            inicio=ruta[0]
            for nodo in ruta[1:]:
                next=nodo
                #solo sumamos si no es el mismo nodo, pues la distancia/tiempo sería 0
                if inicio != next:
                    distancia_recorrida+=value(model.dist[inicio, next])
                    tiempo_total += value(model.tiempo[inicio, next])
                        
                inicio=next
        texto_ruta="-".join(ruta)
        resultados["VehicleId"].append(vehicleid)
        resultados["LoadCap"].append(value(model.cap[vehicleid]))
        resultados["FuelCap"].append(value(model.aut[vehicleid]) / value(model.gas_eff))
        resultados["RouteSequence"].append(texto_ruta)
        resultados["Municipalities"].append(municipios)
        resultados["DemandSatisfied"].append("-".join(demandas_satisfecha))
        resultados["InitialLoad"].append(demanda_total)  # Asumiendo carga inicial igual a demanda satisfecha
        resultados["InitialFuel"].append(distancia_recorrida / value(model.gas_eff))  # Asumiendo tanque lleno
        resultados["Distance"].append(distancia_recorrida)
        resultados["Time"].append(tiempo_total) 
        total_cost = distancia_recorrida * (value(model.gas_cost) / value(model.gas_eff))
        resultados["TotalCost"].append(total_cost)
        
    #guardar dataframe
    data_frame=pd.DataFrame(resultados)
    #export csv
    data_frame.to_csv('caso_base/verificacion_caso1.csv', sep=',', index=False)
        
    

    