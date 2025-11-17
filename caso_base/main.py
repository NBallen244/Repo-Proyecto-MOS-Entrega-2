from pyomo.environ import *
from pyomo.opt import SolverFactory

import sys
import os

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
    #distancias entre nodos
    distancia_dict = {}
    #primero de los centros de distribucion a todos los clientes
    for i in model.D:
        dlongitud= depositos.loc[depositos['StandardizedID'] == i, 'Longitude'].values[0]
        dlatitud= depositos.loc[depositos['StandardizedID'] == i, 'Latitude'].values[0]
        for j in model.C:
            clongitud= clientes.loc[clientes['StandardizedID'] == j, 'Longitude'].values[0]
            clatitud= clientes.loc[clientes['StandardizedID'] == j, 'Latitude'].values[0]
            distancia = ((dlongitud - clongitud)**2 + (dlatitud - clatitud)**2)**0.5
            distancia_dict[(i,j)] = distancia
            distancia_dict[(j,i)] = distancia # Asumiendo simetría
    #luego entre clientes
    for i in model.C:
        clongitud_i= clientes.loc[clientes['StandardizedID'] == i, 'Longitude'].values[0]
        clatitud_i= clientes.loc[clientes['StandardizedID'] == i, 'Latitude'].values[0]
        for j in model.C:
            if i != j:
                clongitud_j= clientes.loc[clientes['StandardizedID'] == j, 'Longitude'].values[0]
                clatitud_j= clientes.loc[clientes['StandardizedID'] == j, 'Latitude'].values[0]
                distancia = ((clongitud_i - clongitud_j)**2 + (clatitud_i - clatitud_j)**2)**0.5
                distancia_dict[(i,j)] = distancia
    model.dist=Param(model.N, model.N, initialize=distancia_dict, default=0)
    #Demandas de clientes
    model.demand=Param(model.C, initialize=clientes.set_index('StandardizedID')['Demand'].to_dict())
    #Capacidad de los vehículos
    model.cap=Param(model.V, initialize=vehiculos.set_index('StandardizedID')['Capacity'].to_dict())
    #Autonomia de los vehículos
    model.aut=Param(model.V, initialize=vehiculos.set_index('StandardizedID')['Range'].to_dict())
    
    return model

if __name__ == "__main__":
    # Cargar los datos
    clientes, depositos, parametros, vehiculos = cargar_datos()
    # Construir el modelo
    model = construccion_modelo(clientes, depositos, parametros, vehiculos)

    