from pyomo.environ import *
from pyomo.opt import SolverFactory

import sys
import os

# Add the directory containing cargaDatos.py to Python's search path
script_dir = os.path.dirname(os.path.abspath("cargaDatos.py"))
sys.path.insert(0, script_dir)

from carga_datos.cargaDatos import cargar_datos_caso2 as cargar_datos

if __name__ == "__main__":
    # Cargar los datos
    clientes, depositos, parametros, vehiculos, estaciones = cargar_datos()

    # Aquí iría el resto del código para definir y resolver el modelo de optimización
    print(depositos)