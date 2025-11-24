Proyecto_LogistiCo/
│
├── carga_datos/
│   ├── cargaDatos.py                 # Carga centralizada de datos para todos los casos
│   ├── project_c /                  # insumos del caso 3
│   └── Proyecto_Caso_Base /         # insumos del caso base
│
├── herramientas_compartidas/
│   ├── distancia.py                 # OSRM + algoritmo Haversine
│   └── README.txt
│
├── caso_base/                       # Solución inicial sin repostaje ni peajes
│   ├── gen_matriz_distancia.py
│   ├── main.py
│   ├── matriz.csv
│   ├── verificacion_caso1.csv
│   └── visuales_resultados.py
│
├── caso_2/                          # Reabastecimiento estratégico en estaciones
│   ├── gen_matriz_distancia.py
│   ├── main.py
│   ├── matriz.csv
│   ├── verificacion_caso2.csv
│   ├── visualizaciones_caso2/       # Mapas y gráficas generadas automáticamente
│   └── visuales_resultados.py
│
├── caso_3/                          # Peso dinámico, peajes y restricciones municipales
│   ├── gen_matriz_distancia.py
│   ├── main.py
│   ├── matriz.csv
│   ├── verificacion_caso3.csv
│   ├── visualizaciones_caso3/       # Gráficas de impacto y sensibilidad
│   └── visuales_resultados.py
│
└── .vscode/                         # Configuración útil del entorno VSCode

