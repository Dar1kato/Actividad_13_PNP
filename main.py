"""
Backend FastAPI: algoritmos sobre grafos ponderados clasificados como P o NP.

Endpoints nuevos:
  POST /dijkstra          -> Dijkstra (P, determinista)
  POST /tsp/exacto        -> Agente viajero exacto (NP-difícil)
                             modo "determinista"    : búsqueda exhaustiva con poda
                             modo "no_determinista" : simulación de "adivinar + verificar"
  POST /tsp/voraz         -> Agente viajero con heurística voraz (vecino más cercano, P)
  POST /grafos/generar    -> Generador de grafos sintéticos ponderados

Ejecución:
  uv add fastapi "uvicorn[standard]"
  uv run uvicorn main:app --reload
  Probar en http://127.0.0.1:8000/docs
"""

import heapq
import math
import random
import threading
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Backend P vs NP sobre grafos",
    version="2.0.0",
    description=(
        "Dijkstra, Agente Viajero (exacto y voraz) y generador de grafos sintéticos. "
        "Cada endpoint devuelve tiempo (ms), memoria pico (KB), nodos recorridos y costo."
    ),
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# El TSP exacto es O(n!) en el peor caso; se limita para no bloquear el servidor.
MAX_NODOS_TSP_EXACTO = 12

# tracemalloc es global al proceso: se serializa la medición entre peticiones.
_candado_medicion = threading.Lock()

EJEMPLO_GRAFO = {
    "A": [["B", 15], ["C", 22], ["D", 10]],
    "B": [["E", 35], ["F", 18]],
    "C": [["E", 20], ["G", 45]],
    "D": [["F", 28], ["G", 30]],
    "E": [["H", 12], ["I", 25]],
    "F": [["H", 40], ["I", 15]],
    "G": [["I", 10], ["J", 50]],
    "H": [["J", 14]],
    "I": [["J", 8]],
    "J": [],
}

# Para el TSP el grafo necesita al menos un ciclo hamiltoniano (el ejemplo
# anterior es un DAG, así que no sirve para el agente viajero).
EJEMPLO_GRAFO_TSP = {
    "A": [["B", 12], ["C", 10], ["D", 19], ["E", 8]],
    "B": [["A", 12], ["C", 3], ["D", 7], ["E", 2]],
    "C": [["A", 10], ["B", 3], ["D", 6], ["E", 20]],
    "D": [["A", 19], ["B", 7], ["C", 6], ["E", 4]],
    "E": [["A", 8], ["B", 2], ["C", 20], ["D", 4]],
}

# ---------------------------------------------------------------------------
# Modelos de entrada y salida
# ---------------------------------------------------------------------------


class GrafoRequest(BaseModel):
    grafo: Dict[str, List[Tuple[str, float]]]


class DijkstraRequest(GrafoRequest):
    origen: str
    destino: str

    model_config = {
        "json_schema_extra": {
            "example": {"grafo": EJEMPLO_GRAFO, "origen": "A", "destino": "J"}
        }
    }


class TSPVorazRequest(GrafoRequest):
    origen: Optional[str] = Field(
        None, description="Nodo de inicio y cierre del ciclo. Por defecto, el primer nodo."
    )

    model_config = {
        "json_schema_extra": {"example": {"grafo": EJEMPLO_GRAFO_TSP, "origen": "A"}}
    }


class TSPExactoRequest(TSPVorazRequest):
    modo: Literal["determinista", "no_determinista"] = "determinista"
    intentos: int = Field(
        10000, ge=1, le=1_000_000,
        description="Solo modo no_determinista: número de recorridos 'adivinados'.",
    )
    semilla: Optional[int] = Field(
        None, description="Solo modo no_determinista: semilla para reproducibilidad."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "grafo": EJEMPLO_GRAFO_TSP,
                "origen": "A",
                "modo": "determinista",
                "intentos": 10000,
                "semilla": 42,
            }
        }
    }


class GenerarGrafoRequest(BaseModel):
    n_nodos: int = Field(10, ge=2, le=200)
    densidad: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Probabilidad de crear cada arista posible adicional (0 a 1).",
    )
    peso_min: float = Field(1, ge=0)
    peso_max: float = Field(50, ge=0)
    dirigido: bool = True
    pesos_enteros: bool = True
    garantizar_ciclo: bool = Field(
        True,
        description="Incluye un ciclo hamiltoniano: el grafo es fuertemente conexo "
        "y siempre admite solución para Dijkstra y TSP.",
    )
    semilla: Optional[int] = None


class Clasificacion(BaseModel):
    clase: str
    complejidad_temporal: str
    complejidad_espacial: str
    comportamiento: str


class ResultadoAlgoritmo(BaseModel):
    algoritmo: str
    clasificacion: Clasificacion
    camino: List[str]
    costo: float
    tiempo_ms: float
    memoria_kb: float
    nodos_recorridos: int
    nodos_explorados: int
    detalle: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


class ErrorGrafo(Exception):
    """Error de entrada o de ejecución que se traduce a HTTP 422."""


Adyacencia = Dict[str, Dict[str, float]]


def normalizar(grafo_crudo: Dict[str, List[Tuple[str, float]]]) -> Adyacencia:
    """Convierte el formato de entrada a {u: {v: peso}}.

    - Agrega nodos que solo aparecen como destino.
    - Si hay aristas duplicadas u->v, conserva la de menor peso.
    - Rechaza pesos negativos o no finitos.
    """
    if not grafo_crudo:
        raise ErrorGrafo("El grafo está vacío.")
    grafo: Adyacencia = {}
    for u, aristas in grafo_crudo.items():
        grafo.setdefault(u, {})
        for v, w in aristas:
            if not math.isfinite(w) or w < 0:
                raise ErrorGrafo(f"Peso inválido en la arista {u}->{v}: {w}. Deben ser >= 0.")
            grafo.setdefault(v, {})
            if v not in grafo[u] or w < grafo[u][v]:
                grafo[u][v] = float(w)
    return grafo


def validar_nodo(grafo: Adyacencia, nodo: str) -> None:
    if nodo not in grafo:
        raise ErrorGrafo(f"El nodo '{nodo}' no existe en el grafo.")


def medir(funcion: Callable[[], Dict[str, Any]]) -> Tuple[Dict[str, Any], float, float]:
    """Ejecuta `funcion` dos veces para no contaminar las métricas:
      1) sin tracemalloc  -> tiempo de ejecución real (ms)
      2) con tracemalloc  -> memoria pico asignada durante el algoritmo (KB)
    Los algoritmos son deterministas (o usan semilla fija), así que ambas
    pasadas producen el mismo resultado.
    """
    with _candado_medicion:
        inicio = time.perf_counter()
        resultado = funcion()
        tiempo_ms = (time.perf_counter() - inicio) * 1000

        tracemalloc.start()
        try:
            funcion()
            _, pico = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return resultado, tiempo_ms, pico / 1024


def ejecutar(
    nombre: str,
    clasificacion: Clasificacion,
    funcion: Callable[[], Dict[str, Any]],
    nodos_recorridos: Callable[[Dict[str, Any]], int],
) -> ResultadoAlgoritmo:
    try:
        r, tiempo_ms, memoria_kb = medir(funcion)
    except ErrorGrafo as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ResultadoAlgoritmo(
        algoritmo=nombre,
        clasificacion=clasificacion,
        camino=r["camino"],
        costo=round(r["costo"], 4),
        tiempo_ms=round(tiempo_ms, 4),
        memoria_kb=round(memoria_kb, 3),
        nodos_recorridos=nodos_recorridos(r),
        nodos_explorados=r["explorados"],
        detalle=r.get("detalle", {}),
    )


# ---------------------------------------------------------------------------
# Algoritmos
# ---------------------------------------------------------------------------


def _dijkstra(grafo: Adyacencia, origen: str, destino: str) -> Dict[str, Any]:
    dist: Dict[str, float] = {origen: 0.0}
    previo: Dict[str, str] = {}
    visitados: set = set()
    cola: List[Tuple[float, str]] = [(0.0, origen)]
    explorados = 0  # nodos extraídos (fijados) de la cola de prioridad

    while cola:
        d, u = heapq.heappop(cola)
        if u in visitados:
            continue
        visitados.add(u)
        explorados += 1
        if u == destino:
            break
        for v, w in grafo[u].items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                previo[v] = u
                heapq.heappush(cola, (nd, v))

    if destino not in visitados:
        raise ErrorGrafo(f"No existe camino de '{origen}' a '{destino}'.")

    camino = [destino]
    while camino[-1] != origen:
        camino.append(previo[camino[-1]])
    camino.reverse()
    return {"camino": camino, "costo": dist[destino], "explorados": explorados}


def _tsp_determinista(grafo: Adyacencia, origen: str) -> Dict[str, Any]:
    """Búsqueda exhaustiva (DFS) con poda por costo: O(n!) en el peor caso."""
    n = len(grafo)
    if n == 1:
        return {"camino": [origen, origen], "costo": 0.0, "explorados": 1}

    mejor_costo = math.inf
    mejor_ruta: Optional[List[str]] = None
    explorados = 0
    ruta = [origen]
    visitados = {origen}

    def dfs(u: str, costo: float) -> None:
        nonlocal mejor_costo, mejor_ruta, explorados
        explorados += 1
        if costo >= mejor_costo:  # poda
            return
        if len(ruta) == n:
            cierre = grafo[u].get(origen)
            if cierre is not None and costo + cierre < mejor_costo:
                mejor_costo = costo + cierre
                mejor_ruta = ruta + [origen]
            return
        for v, w in sorted(grafo[u].items(), key=lambda a: a[1]):
            if v not in visitados:
                visitados.add(v)
                ruta.append(v)
                dfs(v, costo + w)
                ruta.pop()
                visitados.remove(v)

    dfs(origen, 0.0)
    if mejor_ruta is None:
        raise ErrorGrafo("El grafo no tiene ningún ciclo hamiltoniano desde el origen.")
    return {"camino": mejor_ruta, "costo": mejor_costo, "explorados": explorados}


def _verificar_certificado(grafo: Adyacencia, ruta: List[str]) -> Tuple[Optional[float], int]:
    """Verificador polinomial O(n): ¿la ruta es un ciclo válido? Devuelve (costo|None, pasos)."""
    costo = 0.0
    pasos = 0
    for a, b in zip(ruta, ruta[1:]):
        pasos += 1
        w = grafo[a].get(b)
        if w is None:
            return None, pasos
        costo += w
    return costo, pasos


def _tsp_no_determinista(
    grafo: Adyacencia, origen: str, intentos: int, semilla: Optional[int]
) -> Dict[str, Any]:
    """Simula el modelo no determinista 'adivinar y verificar'.

    Una máquina no determinista adivinaría el recorrido óptimo en una rama y lo
    verificaría en tiempo polinomial. En una máquina determinista solo se puede
    simular: se adivinan `intentos` permutaciones al azar y cada una se verifica
    en O(n). No garantiza encontrar el óptimo (ni siquiera uno válido).
    """
    n = len(grafo)
    if n == 1:
        return {"camino": [origen, origen], "costo": 0.0, "explorados": 1}

    rng = random.Random(semilla)
    otros = [x for x in grafo if x != origen]
    mejor_costo = math.inf
    mejor_ruta: Optional[List[str]] = None
    explorados = 0
    validos = 0

    for _ in range(intentos):
        rng.shuffle(otros)  # "adivinar" el certificado
        ruta = [origen] + otros + [origen]
        costo, pasos = _verificar_certificado(grafo, ruta)  # verificar en O(n)
        explorados += pasos
        if costo is not None:
            validos += 1
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_ruta = ruta[:]

    if mejor_ruta is None:
        raise ErrorGrafo(
            f"Ninguno de los {intentos} recorridos adivinados fue un ciclo válido. "
            "Aumenta 'intentos' o usa el modo determinista."
        )
    return {
        "camino": mejor_ruta,
        "costo": mejor_costo,
        "explorados": explorados,
        "detalle": {
            "intentos": intentos,
            "intentos_validos": validos,
            "garantiza_optimo": False,
        },
    }


def _tsp_voraz(grafo: Adyacencia, origen: str) -> Dict[str, Any]:
    """Vecino más cercano: en cada paso va a la arista más barata hacia un nodo no visitado."""
    n = len(grafo)
    ruta = [origen]
    visitados = {origen}
    costo = 0.0
    actual = origen
    explorados = 1

    while len(ruta) < n:
        candidatos = [(w, v) for v, w in grafo[actual].items() if v not in visitados]
        if not candidatos:
            raise ErrorGrafo(
                f"La heurística voraz quedó atrapada en '{actual}' sin nodos no visitados "
                "alcanzables (el grafo no es completo). Puede existir un ciclo que la "
                "heurística no encontró."
            )
        w, v = min(candidatos)
        costo += w
        ruta.append(v)
        visitados.add(v)
        actual = v
        explorados += 1

    cierre = grafo[actual].get(origen)
    if cierre is None and n > 1:
        raise ErrorGrafo(f"No existe arista de regreso de '{actual}' al origen '{origen}'.")
    if n > 1:
        costo += cierre
    ruta.append(origen)
    return {"camino": ruta, "costo": costo, "explorados": explorados}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

CLASIF_DIJKSTRA = Clasificacion(
    clase="P",
    complejidad_temporal="O((V + E) log V) con cola de prioridad",
    complejidad_espacial="O(V)",
    comportamiento=(
        "Determinista: para la misma entrada siempre sigue la misma secuencia de "
        "pasos y devuelve el camino óptimo. Se resuelve en tiempo polinomial; "
        "no requiere no determinismo."
    ),
)

CLASIF_TSP_DET = Clasificacion(
    clase="NP-difícil (versión de decisión: NP-completo)",
    complejidad_temporal="O(n!) en el peor caso (con poda)",
    complejidad_espacial="O(n)",
    comportamiento=(
        "Determinista: explora el espacio de recorridos de forma exhaustiva y "
        "garantiza el óptimo, pero el tiempo crece factorialmente con n."
    ),
)

CLASIF_TSP_NODET = Clasificacion(
    clase="NP (verificación polinomial)",
    complejidad_temporal=(
        "Modelo no determinista: O(n) (adivinar + verificar). "
        "Simulación: O(intentos · n)"
    ),
    complejidad_espacial="O(n)",
    comportamiento=(
        "No determinista simulado: una máquina no determinista adivinaría el "
        "recorrido en una rama y lo verificaría en tiempo polinomial. Aquí se "
        "simula con recorridos aleatorios: no garantiza el óptimo."
    ),
)

CLASIF_TSP_VORAZ = Clasificacion(
    clase="P (heurística polinomial para un problema NP-difícil)",
    complejidad_temporal="O(n²)",
    complejidad_espacial="O(n)",
    comportamiento=(
        "Determinista: elige siempre la arista más barata disponible. Es rápido "
        "pero no garantiza el óptimo y puede fallar en grafos no completos."
    ),
)


@app.post("/dijkstra", response_model=ResultadoAlgoritmo, tags=["Algoritmos"])
def endpoint_dijkstra(req: DijkstraRequest):
    """Camino de costo mínimo entre `origen` y `destino` (pesos >= 0)."""
    try:
        grafo = normalizar(req.grafo)
        validar_nodo(grafo, req.origen)
        validar_nodo(grafo, req.destino)
    except ErrorGrafo as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ejecutar(
        "Dijkstra",
        CLASIF_DIJKSTRA,
        lambda: _dijkstra(grafo, req.origen, req.destino),
        nodos_recorridos=lambda r: len(r["camino"]),
    )


@app.post("/tsp/exacto", response_model=ResultadoAlgoritmo, tags=["Algoritmos"])
def endpoint_tsp_exacto(req: TSPExactoRequest):
    """Agente viajero exacto. Busca el ciclo hamiltoniano de costo mínimo."""
    try:
        grafo = normalizar(req.grafo)
        origen = req.origen or next(iter(grafo))
        validar_nodo(grafo, origen)
        if req.modo == "determinista" and len(grafo) > MAX_NODOS_TSP_EXACTO:
            raise ErrorGrafo(
                f"El TSP exacto determinista está limitado a {MAX_NODOS_TSP_EXACTO} nodos "
                f"(recibidos: {len(grafo)}) porque es O(n!). Usa /tsp/voraz o el modo no_determinista."
            )
    except ErrorGrafo as e:
        raise HTTPException(status_code=422, detail=str(e))

    if req.modo == "determinista":
        return ejecutar(
            "TSP exacto (determinista)",
            CLASIF_TSP_DET,
            lambda: _tsp_determinista(grafo, origen),
            nodos_recorridos=lambda r: len(r["camino"]) - 1,
        )
    return ejecutar(
        "TSP (no determinista simulado)",
        CLASIF_TSP_NODET,
        lambda: _tsp_no_determinista(grafo, origen, req.intentos, req.semilla),
        nodos_recorridos=lambda r: len(r["camino"]) - 1,
    )


@app.post("/tsp/voraz", response_model=ResultadoAlgoritmo, tags=["Algoritmos"])
def endpoint_tsp_voraz(req: TSPVorazRequest):
    """Agente viajero con heurística voraz (vecino más cercano)."""
    try:
        grafo = normalizar(req.grafo)
        origen = req.origen or next(iter(grafo))
        validar_nodo(grafo, origen)
    except ErrorGrafo as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ejecutar(
        "TSP voraz (vecino más cercano)",
        CLASIF_TSP_VORAZ,
        lambda: _tsp_voraz(grafo, origen),
        nodos_recorridos=lambda r: len(r["camino"]) - 1,
    )


def _nombre_nodo(i: int) -> str:
    """0->A, 25->Z, 26->AA, 27->AB ..."""
    i += 1
    s = ""
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


@app.post("/grafos/generar", tags=["Auxiliar"])
def generar_grafo(req: GenerarGrafoRequest):
    """Genera un grafo sintético ponderado en el mismo formato de entrada.

    Copia el objeto `grafo` de la respuesta y pégalo en los demás endpoints.
    """
    if req.peso_min > req.peso_max:
        raise HTTPException(status_code=422, detail="peso_min no puede ser mayor que peso_max.")

    rng = random.Random(req.semilla)
    if req.pesos_enteros:
        lo, hi = math.ceil(req.peso_min), math.floor(req.peso_max)
        if lo > hi:
            raise HTTPException(
                status_code=422, detail="No hay enteros en el rango [peso_min, peso_max]."
            )

        def peso() -> float:
            return rng.randint(lo, hi)
    else:

        def peso() -> float:
            return round(rng.uniform(req.peso_min, req.peso_max), 2)

    nombres = [_nombre_nodo(i) for i in range(req.n_nodos)]
    ady: Dict[str, Dict[str, float]] = {n: {} for n in nombres}

    def agregar(u: str, v: str) -> None:
        if u == v or v in ady[u]:
            return
        w = peso()
        ady[u][v] = w
        if not req.dirigido:
            ady[v][u] = w

    if req.garantizar_ciclo:
        orden = nombres[:]
        rng.shuffle(orden)
        for i in range(req.n_nodos):
            agregar(orden[i], orden[(i + 1) % req.n_nodos])

    for i, u in enumerate(nombres):
        for j, v in enumerate(nombres):
            if i == j or (not req.dirigido and j < i):
                continue
            if rng.random() < req.densidad:
                agregar(u, v)

    grafo = {u: [[v, w] for v, w in aristas.items()] for u, aristas in ady.items()}
    return {
        "grafo": grafo,
        "metadatos": {
            "n_nodos": req.n_nodos,
            "n_aristas_dirigidas": sum(len(a) for a in ady.values()),
            "dirigido": req.dirigido,
            "ciclo_hamiltoniano_garantizado": req.garantizar_ciclo,
            "semilla": req.semilla,
        },
    }