"""
DevFilter IA - Fase 2 del Proyecto Integrador (Inteligencia Artificial, UMNG)
Caso C: Preselección automatizada de talento tecnológico.

Problema de búsqueda: de N candidatos sintéticos evaluados en orden fijo,
construir la lista de preselección de exactamente K candidatos que maximice
la utilidad total U(C, V) (equivalente a minimizar el costo de camino).

Implementa BFS, DFS y A* con la misma interfaz de la clase Problem de
AIMA-Python (initial, actions, result, goal_test, path_cost, h).
Uso:  python devfilter_busqueda.py
"""
import heapq
import time
from collections import deque

# ---------------------------------------------------------------------------
# 0. Clases base (misma interfaz que aima-python/search.py)
# ---------------------------------------------------------------------------
class Problem:
    """Clase abstracta de problema de búsqueda (interfaz AIMA)."""

    def __init__(self, initial, goal=None):
        self.initial, self.goal = initial, goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def h(self, node):
        return 0


class Node:
    """Nodo del árbol de búsqueda (interfaz AIMA)."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state, self.parent = state, parent
        self.action, self.path_cost = action, path_cost

    def expand(self, problem):
        return [self.child_node(problem, a) for a in problem.actions(self.state)]

    def child_node(self, problem, action):
        nxt = problem.result(self.state, action)
        return Node(nxt, self, action,
                    problem.path_cost(self.path_cost, self.state, action, nxt))

    def solution(self):
        nodo, acc = self, []
        while nodo.parent is not None:
            acc.append(nodo.action)
            nodo = nodo.parent
        return acc[::-1]

    def __lt__(self, other):          # desempate en la cola de prioridad
        return self.path_cost < other.path_cost


# ---------------------------------------------------------------------------
# 1. DATOS SINTÉTICOS Y FUNCIÓN DE UTILIDAD U(C, V)
# ---------------------------------------------------------------------------
PESOS_OBLIGATORIAS = {"html_css": 15, "javascript": 20, "git": 10}   # 45 pts
PESOS_DESEABLES = {"react": 10, "node": 5, "sql": 5}                 # 20 pts
# + calidad de repositorios (0-20) + prueba técnica (0-15)  => máximo 100

# (id, tecnologías, calidad_repos, prueba_técnica)
CANDIDATOS_BRUTOS = [
    ("C01", {"html_css", "javascript", "git", "react", "sql"}, 16, 12),
    ("C02", {"html_css", "javascript", "git"}, 10, 9),
    ("C03", {"html_css", "javascript", "git", "react", "node", "sql"}, 18, 14),
    ("C04", {"html_css", "javascript"}, 12, 10),          # sin Git: no elegible
    ("C05", {"html_css", "javascript", "git", "node"}, 14, 11),
    ("C06", {"html_css", "javascript", "git", "react"}, 9, 13),
    ("C07", {"html_css", "git", "sql"}, 15, 8),            # sin JS: no elegible
    ("C08", {"html_css", "javascript", "git", "react", "node"}, 17, 15),
    ("C09", {"html_css", "javascript", "git", "sql"}, 11, 7),
    ("C10", {"html_css", "javascript", "git", "react", "node", "sql"}, 8, 10),
    ("C11", {"html_css", "javascript", "git"}, 13, 12),
    ("C12", {"javascript", "git", "react"}, 19, 14),       # sin HTML/CSS: no elegible
    ("C13", {"html_css", "javascript", "git", "node", "sql"}, 12, 9),
    ("C14", {"html_css", "javascript", "git", "react"}, 15, 13),
]


def utilidad(techs, repos, prueba):
    """U(C, V) discreta en [0, 100]."""
    u = sum(p for t, p in PESOS_OBLIGATORIAS.items() if t in techs)
    u += sum(p for t, p in PESOS_DESEABLES.items() if t in techs)
    return u + repos + prueba


def es_elegible(techs):
    """Cumple todos los requisitos mínimos (tecnologías obligatorias)."""
    return all(t in techs for t in PESOS_OBLIGATORIAS)


CANDIDATOS = [{"id": n, "U": utilidad(t, r, p), "elegible": es_elegible(t)}
              for n, t, r, p in CANDIDATOS_BRUTOS]
N, K, U_MAX = len(CANDIDATOS), 4, 100


# ---------------------------------------------------------------------------
# 2. FORMALIZACIÓN DEL ESPACIO DE ESTADOS
# ---------------------------------------------------------------------------
class PreseleccionProblem(Problem):
    """
    Estado s = (i, S): i = siguiente candidato a evaluar,
                       S = tupla de índices ya preseleccionados.
    s0 = (0, ())
    ACTIONS(s): 'ACEPTAR' (si el candidato i es elegible y |S| < K), 'RECHAZAR'
    RESULT:     ACEPTAR -> (i+1, S+(i,));  RECHAZAR -> (i+1, S)
    IS-GOAL(s): i == N  y  |S| == K
    c(s,a,s'):  ACEPTAR = U_MAX - U(c_i);  RECHAZAR = 0
    """

    def __init__(self):
        super().__init__(initial=(0, ()))

    def actions(self, state):
        i, S = state
        if i >= N:
            return []
        acc = []
        if CANDIDATOS[i]["elegible"] and len(S) < K:
            acc.append("ACEPTAR")
        acc.append("RECHAZAR")
        return acc

    def result(self, state, action):
        i, S = state
        return (i + 1, S + (i,)) if action == "ACEPTAR" else (i + 1, S)

    def goal_test(self, state):
        i, S = state
        return i == N and len(S) == K

    def path_cost(self, c, state1, action, state2):
        if action == "ACEPTAR":
            return c + U_MAX - CANDIDATOS[state1[0]]["U"]
        return c

    def h(self, node):
        """Heurística ADMISIBLE (y consistente): suma de los (K-|S|) menores
        costos de aceptación entre los candidatos elegibles pendientes."""
        i, S = node.state
        faltan = K - len(S)
        if faltan <= 0:
            return 0
        costos = sorted(U_MAX - c["U"] for c in CANDIDATOS[i:] if c["elegible"])
        if len(costos) < faltan:
            return float("inf")            # estado sin solución posible
        return sum(costos[:faltan])

    def h_no_admisible(self, node, w=2.0):
        """h' = w*h (w > 1) sobreestima el costo: A* pierde la optimalidad."""
        h = self.h(node)
        return h if h == float("inf") else w * h


# ---------------------------------------------------------------------------
# 3. ALGORITMOS DE BÚSQUEDA (instrumentados con métricas)
# ---------------------------------------------------------------------------
def _salida(nodo, expandidos, max_frontera, t0):
    return {"nodo": nodo, "costo": nodo.path_cost if nodo else None,
            "expandidos": expandidos, "max_frontera": max_frontera,
            "tiempo_ms": (time.perf_counter() - t0) * 1000}


def breadth_first_search(problem):
    """BFS: cola FIFO. Prueba de meta al extraer el nodo."""
    t0 = time.perf_counter()
    frontera, exp, mx = deque([Node(problem.initial)]), 0, 1
    while frontera:
        nodo = frontera.popleft()
        if problem.goal_test(nodo.state):
            return _salida(nodo, exp, mx, t0)
        exp += 1
        frontera.extend(nodo.expand(problem))
        mx = max(mx, len(frontera))
    return _salida(None, exp, mx, t0)


def depth_first_search(problem):
    """DFS: pila LIFO. Explora primero la acción ACEPTAR."""
    t0 = time.perf_counter()
    frontera, exp, mx = [Node(problem.initial)], 0, 1
    while frontera:
        nodo = frontera.pop()
        if problem.goal_test(nodo.state):
            return _salida(nodo, exp, mx, t0)
        exp += 1
        frontera.extend(reversed(nodo.expand(problem)))
        mx = max(mx, len(frontera))
    return _salida(None, exp, mx, t0)


def astar_search(problem, h=None):
    """A*: expande el nodo de menor f(n) = g(n) + h(n)."""
    h = h or problem.h
    t0, cont = time.perf_counter(), 0
    raiz = Node(problem.initial)
    frontera, exp, mx = [(h(raiz), cont, raiz)], 0, 1
    while frontera:
        _, _, nodo = heapq.heappop(frontera)
        if problem.goal_test(nodo.state):
            return _salida(nodo, exp, mx, t0)
        exp += 1
        for hijo in nodo.expand(problem):
            hh = h(hijo)
            if hh == float("inf"):
                continue                    # poda de estados sin solución
            cont += 1
            heapq.heappush(frontera, (hijo.path_cost + hh, cont, hijo))
        mx = max(mx, len(frontera))
    return _salida(None, exp, mx, t0)


# ---------------------------------------------------------------------------
# 4. EXPERIMENTOS Y BENCHMARK
# ---------------------------------------------------------------------------
def lista_final(res):
    return [CANDIDATOS[i]["id"] for i in res["nodo"].state[1]]


if __name__ == "__main__":
    print("Candidatos (U = utilidad):")
    for c in CANDIDATOS:
        print(f"  {c['id']}  U={c['U']:3d}  elegible={c['elegible']}")
    print(f"\nN = {N} candidatos, K = {K} cupos\n")

    p = PreseleccionProblem()
    corridas = [
        ("BFS", lambda: breadth_first_search(p)),
        ("DFS", lambda: depth_first_search(p)),
        ("A* (h admisible)", lambda: astar_search(p)),
        ("A* (h no admisible, w=2)", lambda: astar_search(p, p.h_no_admisible)),
    ]
    REPS = 30   # el tiempo se reporta como mediana de REPS ejecuciones
    print(f"{'Algoritmo':<26}{'Expand.':>9}{'Frontera':>10}"
          f"{'Tiempo(ms)':>12}{'Costo':>7}{'U total':>9}  Lista")
    for nombre, f in corridas:
        tiempos = []
        for _ in range(REPS):
            r = f()
            tiempos.append(r["tiempo_ms"])
        t_med = sorted(tiempos)[REPS // 2]
        print(f"{nombre:<26}{r['expandidos']:>9}{r['max_frontera']:>10}"
              f"{t_med:>12.3f}{r['costo']:>7}{K*U_MAX - r['costo']:>9}"
              f"  {lista_final(r)}")
