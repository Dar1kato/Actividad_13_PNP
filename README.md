# Backend P vs NP sobre grafos ponderados (FastAPI + uv)

**Autor:** Daniel Peña Cruz - 200818

Backend en Python con FastAPI que implementa algoritmos sobre grafos ponderados, los clasifica como **P** o **NP** según su complejidad y muestra su comportamiento en escenarios **deterministas** y **no deterministas**. Todo el código vive en un único archivo, `main.py`, dentro de un proyecto gestionado con `uv`.

## Endpoints

| Método | Ruta | Descripción | Clase |
|---|---|---|---|
| POST | `/dijkstra` | Camino de costo mínimo entre `origen` y `destino` | P, O((V+E) log V) |
| POST | `/tsp/exacto` | Agente viajero exacto (`modo`: `determinista` o `no_determinista`) | NP-difícil, O(n!) |
| POST | `/tsp/voraz` | Agente viajero con heurística voraz (vecino más cercano) | P, O(n²) |
| POST | `/grafos/generar` | Genera grafos sintéticos ponderados | Auxiliar |

## Formato del grafo

Lista de adyacencia: cada nodo apunta a una lista de pares `[vecino, peso]`.

```json
{
  "grafo": {
    "A": [["B", 15], ["C", 22], ["D", 10]],
    "B": [["E", 35], ["F", 18]],
    "C": [["E", 20], ["G", 45]],
    "D": [["F", 28], ["G", 30]],
    "E": [["H", 12], ["I", 25]],
    "F": [["H", 40], ["I", 15]],
    "G": [["I", 10], ["J", 50]],
    "H": [["J", 14]],
    "I": [["J", 8]],
    "J": []
  }
}
```

- Los grafos se interpretan como **dirigidos**.
- Los pesos deben ser finitos y ≥ 0.
- Los nodos que solo aparecen como destino se agregan automáticamente.
- Si hay aristas duplicadas `u -> v`, se conserva la de menor peso.

## Información devuelta

Los tres algoritmos responden con la misma estructura:

| Campo | Significado |
|---|---|
| `algoritmo` | Nombre del algoritmo ejecutado |
| `clasificacion` | Clase (P / NP), complejidad temporal y espacial, y descripción del comportamiento determinista o no determinista |
| `camino` | Secuencia de nodos de la solución (en el TSP incluye el regreso al origen) |
| `costo` | Costo total del camino |
| `tiempo_ms` | Tiempo de ejecución en milisegundos |
| `memoria_kb` | Pico de memoria asignada durante el algoritmo, en KB |
| `nodos_recorridos` | Nodos del camino resultante (en el TSP, nodos distintos visitados) |
| `nodos_explorados` | Nodos que el algoritmo tocó durante la búsqueda |
| `detalle` | Información adicional (p. ej. intentos válidos en el modo no determinista) |

### Cómo se miden tiempo y memoria

La función `medir()` ejecuta el algoritmo **dos veces**:

1. Sin `tracemalloc`, para obtener un tiempo real con `time.perf_counter()`.
2. Con `tracemalloc`, para obtener el pico de memoria, sin que la instrumentación altere el tiempo.

Los algoritmos son deterministas (o usan semilla fija), así que ambas pasadas dan el mismo resultado. La medición se serializa con un candado porque `tracemalloc` es global al proceso. La memoria reportada **no** incluye el grafo ya construido, solo lo que el algoritmo asigna al ejecutarse.

## Funcionamiento de cada algoritmo

### Dijkstra (`/dijkstra`)
Usa una cola de prioridad (`heapq`). Extrae el nodo con menor distancia acumulada, lo fija y relaja sus aristas. Se detiene al fijar el destino y reconstruye el camino con los predecesores. Es **determinista y polinomial (P)**: no necesita no determinismo. Devuelve 422 si no existe camino.

### TSP exacto (`/tsp/exacto`)
Busca el ciclo hamiltoniano de costo mínimo que parte y regresa al `origen`.

- **`modo: "determinista"`**: búsqueda en profundidad exhaustiva con poda (se abandona una rama si su costo parcial ya supera el mejor ciclo encontrado). Garantiza el óptimo, pero es O(n!) en el peor caso, por eso está limitado a **12 nodos** (`MAX_NODOS_TSP_EXACTO`).
- **`modo: "no_determinista"`**: simula el esquema *adivinar y verificar* que define a la clase NP. Una máquina no determinista adivinaría el recorrido óptimo y lo verificaría en O(n). Como no existe tal máquina, se simula generando `intentos` permutaciones aleatorias (reproducibles con `semilla`) y verificando cada una en O(n). **No garantiza el óptimo** ni encontrar un ciclo válido.

### TSP voraz (`/tsp/voraz`)
Heurística del vecino más cercano: desde el origen, avanza siempre por la arista más barata hacia un nodo no visitado y al final cierra el ciclo. Es **polinomial (O(n²))** pero no garantiza el óptimo. En grafos no completos puede quedar atrapado en un nodo sin salida; en ese caso responde 422 explicando el motivo.

### Generador de grafos (`/grafos/generar`)
Crea un grafo con nombres de nodo `A, B, ..., Z, AA, AB, ...`. Parámetros:

| Parámetro | Descripción | Por defecto |
|---|---|---|
| `n_nodos` | Número de nodos (2 a 200) | 10 |
| `densidad` | Probabilidad de cada arista adicional (0 a 1) | 0.3 |
| `peso_min`, `peso_max` | Rango de pesos | 1, 50 |
| `dirigido` | Grafo dirigido o no dirigido | `true` |
| `pesos_enteros` | Pesos enteros o decimales | `true` |
| `garantizar_ciclo` | Incluye un ciclo hamiltoniano, de modo que el grafo es fuertemente conexo y el TSP tiene solución | `true` |
| `semilla` | Semilla para reproducibilidad | aleatoria |

La respuesta trae `grafo` (listo para copiarse a los demás endpoints) y `metadatos`.

## Notas sobre los grafos de prueba

El grafo de ejemplo del formato es un **DAG** (el nodo `J` no tiene salidas). Funciona con Dijkstra, pero **no** con el TSP, que requiere un ciclo hamiltoniano. Para probar el TSP usa el ejemplo que trae `/docs` o genera un grafo con `/grafos/generar`.

## Requisitos

- Python 3.10 o superior
- [uv](https://docs.astral.sh/uv/) instalado
- Dependencias: `fastapi` y `uvicorn[standard]`

## Cómo replicar y correr el backend

1. **Crear el proyecto**
   ```bash
   uv init p-vs-np-grafos
   cd p-vs-np-grafos
   ```
   Si ya tienes la carpeta de un proyecto previo, puedes reutilizarla.

2. **Instalar dependencias**
   ```bash
   uv add fastapi "uvicorn[standard]"
   ```

3. **Colocar el código**
   Copia el archivo `main.py` en la raíz del proyecto, reemplazando el generado por `uv init`.

4. **Ejecutar el servidor**
   ```bash
   uv run uvicorn main:app --reload
   ```

5. **Probar desde `/docs`**
   Abre <http://127.0.0.1:8000/docs>, elige un endpoint, pulsa **Try it out**, pega el JSON y pulsa **Execute**.

## Ejemplos de uso

**Dijkstra**
```bash
curl -X POST http://127.0.0.1:8000/dijkstra \
  -H "Content-Type: application/json" \
  -d '{"grafo": {"A": [["B", 15], ["C", 22], ["D", 10]], "B": [["E", 35], ["F", 18]], "C": [["E", 20], ["G", 45]], "D": [["F", 28], ["G", 30]], "E": [["H", 12], ["I", 25]], "F": [["H", 40], ["I", 15]], "G": [["I", 10], ["J", 50]], "H": [["J", 14]], "I": [["J", 8]], "J": []}, "origen": "A", "destino": "J"}'
```

Resultado esperado: camino `A → B → F → I → J` con costo `56`.

**Generar un grafo**
```bash
curl -X POST http://127.0.0.1:8000/grafos/generar \
  -H "Content-Type: application/json" \
  -d '{"n_nodos": 8, "densidad": 0.4, "semilla": 3}'
```

**TSP exacto no determinista**
```json
{
  "grafo": {
    "A": [["B", 12], ["C", 10], ["D", 19], ["E", 8]],
    "B": [["A", 12], ["C", 3], ["D", 7], ["E", 2]],
    "C": [["A", 10], ["B", 3], ["D", 6], ["E", 20]],
    "D": [["A", 19], ["B", 7], ["C", 6], ["E", 4]],
    "E": [["A", 8], ["B", 2], ["C", 20], ["D", 4]]
  },
  "origen": "A",
  "modo": "no_determinista",
  "intentos": 10000,
  "semilla": 42
}
```

## Códigos de error

| Código | Causa |
|---|---|
| 422 | Grafo vacío, pesos negativos, nodo inexistente, camino o ciclo inexistente, más de 12 nodos en el TSP exacto determinista, o la heurística voraz quedó atrapada |