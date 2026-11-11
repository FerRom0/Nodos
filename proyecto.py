"""
Laboratorio de algoritmos de búsqueda - Fundamentos de Inteligencia Artificial (UTNA)
Versión en el estilo visto en clase: clase Nodo, lista de hijos (nodo, costo),
atributo heurística, bandera visitado y prints de "Visitando nodo".

    python laboratorio_busqueda_estilo_clase.py          -> interfaz gráfica (Tkinter)
    python laboratorio_busqueda_estilo_clase.py --cli    -> comparación en consola

    Integrantes:
    Luis Fernando Romo Castorena
    Cesar Emiliano Contreras Ovalle
    Kelvin Bryan Sanchez Ortega
    Jeremy Ramirez Escobar
"""
import heapq
import math
import sys
import time
from collections import deque


# ----------------------------------------------------------------------------
# 1. MODELADO: clase Nodo y construcción del grafo
# ----------------------------------------------------------------------------
class Nodo:
    def __init__(self, value, nombre, x, y):
        self.value = value          # letra del nodo
        self.nombre = nombre        # nombre de la ubicación
        self.x = x                  # coordenadas (km aprox.) para la heurística
        self.y = y
        self.heuristica = 0         # distancia estimada al destino
        self.visitado = False
        self.hijos = []             # lista de (nodo, costo)

    def agregar_hijo(self, nodo, costo):
        self.hijos.append((nodo, costo))


DATOS_NODOS = [
    ("A", "Almacén Central", 5, 50), ("B", "Planta Norte", 20, 80),
    ("C", "Centro de Distribución", 20, 50), ("D", "Puerto Seco", 20, 20),
    ("E", "Bodega Norte", 40, 85), ("F", "Hub Regional", 40, 55),
    ("G", "Terminal Sur", 40, 25), ("H", "Zona Industrial", 60, 75),
    ("I", "Nodo Carretero", 60, 40), ("J", "Aduana", 60, 10),
    ("K", "Plaza Comercial", 80, 60), ("L", "Cliente Final", 95, 50),
    ("M", "Parque Logístico", 78, 48),
]

# (origen, destino, costo, bloqueada)  -> conexiones en ambos sentidos
DATOS_CONEXIONES = [
    ("A", "B", 38, False), ("A", "C", 18, False), ("A", "D", 36, False),
    ("B", "C", 34, False), ("B", "E", 24, False), ("C", "F", 22, False),
    ("C", "D", 32, False), ("D", "G", 23, False), ("E", "H", 26, False),
    ("E", "F", 33, False), ("F", "G", 34, False), ("F", "I", 28, False),
    ("F", "H", 31, False), ("G", "I", 27, False), ("G", "J", 30, False),
    ("H", "K", 29, False), ("I", "K", 30, False), ("J", "I", 33, False),
    ("J", "L", 60, False), ("K", "L", 21, False), ("F", "M", 70, False),
    ("M", "L", 20, False), ("M", "K", 14, False),
    ("F", "K", 42, True), ("I", "L", 40, True),     # conexiones BLOQUEADAS
]

NODOS = {v: Nodo(v, n, x, y) for v, n, x, y in DATOS_NODOS}
COSTOS = {}
for a, b, costo, bloqueada in DATOS_CONEXIONES:
    if not bloqueada:                      # las bloqueadas NO se agregan como hijos
        NODOS[a].agregar_hijo(NODOS[b], costo)
        NODOS[b].agregar_hijo(NODOS[a], costo)
        COSTOS[(a, b)] = COSTOS[(b, a)] = costo
for _n in NODOS.values():
    _n.hijos.sort(key=lambda par: par[0].value)   # orden alfabético (determinista)

INICIO_DEFECTO, DESTINO_DEFECTO = "A", "L"


def reiniciar():
    for n in NODOS.values():
        n.visitado = False


def costo_ruta(ruta):
    return sum(COSTOS[(ruta[i], ruta[i + 1])] for i in range(len(ruta) - 1))


def reconstruir(padres, valor):
    ruta = []
    while valor is not None:
        ruta.append(valor)
        valor = padres[valor]
    return ruta[::-1]


# ----------------------------------------------------------------------------
# Heurísticas (se guardan en el atributo nodo.heuristica)
# ----------------------------------------------------------------------------
def h_euclidiana(n, meta):
    return math.hypot(n.x - meta.x, n.y - meta.y)


def h_manhattan(n, meta):
    return abs(n.x - meta.x) + abs(n.y - meta.y)


def h_exagerada(n, meta):
    return 3 * h_euclidiana(n, meta)       # sobreestima: NO admisible


def h_cero(n, meta):
    return 0


HEURISTICAS = {
    "Euclidiana (admisible)": h_euclidiana,
    "Manhattan": h_manhattan,
    "Euclidiana x3 (no admisible)": h_exagerada,
    "Cero (sin información)": h_cero,
}


def calcular_heuristicas(objetivo, funcion=h_euclidiana):
    for n in NODOS.values():
        n.heuristica = funcion(n, objetivo)


# ----------------------------------------------------------------------------
# 2. ALGORITMOS  (devuelven: ruta, orden de nodos explorados)
# ----------------------------------------------------------------------------
def bfs(inicio, objetivo, verbose=False):
    reiniciar()
    cola = deque([inicio])
    padres = {inicio.value: None}
    orden = []
    while cola:
        actual = cola.popleft()
        orden.append(actual.value)
        if verbose:
            print(len(orden), "Visitando nodo:", actual.value)
        if actual == objetivo:
            return reconstruir(padres, actual.value), orden
        for hijo, _ in actual.hijos:
            if hijo.value not in padres:          # evita repetir nodos
                padres[hijo.value] = actual.value
                cola.append(hijo)
    return [], orden


def dfs(inicio, objetivo, verbose=False):
    reiniciar()
    pila = [inicio]
    inicio.visitado = True
    orden = [inicio.value]
    if verbose:
        print(1, "Visitando nodo:", inicio.value)
    while pila:
        actual = pila[-1]                          # tope de la pila
        if actual == objetivo:
            return [n.value for n in pila], orden
        siguiente = None
        for hijo, _ in actual.hijos:
            if not hijo.visitado:
                siguiente = hijo
                break
        if siguiente is not None:                  # profundidad
            siguiente.visitado = True
            orden.append(siguiente.value)
            if verbose:
                print(len(orden), "Visitando nodo:", siguiente.value)
            pila.append(siguiente)
        else:                                      # retroceso (backtracking)
            pila.pop()
    return [], orden


def greedy(inicio, objetivo, verbose=False):
    reiniciar()
    contador = 0                                   # desempate
    frontera = [(inicio.heuristica, contador, inicio)]
    padres = {inicio.value: None}
    orden = []
    while frontera:
        h, _, actual = heapq.heappop(frontera)
        if actual.visitado:
            continue
        actual.visitado = True
        orden.append(actual.value)
        if verbose:
            print(len(orden), "Visitando nodo:", actual.value, "| h =", round(h, 1))
        if actual == objetivo:
            return reconstruir(padres, actual.value), orden
        for hijo, _ in actual.hijos:
            if not hijo.visitado and hijo.value not in padres:
                padres[hijo.value] = actual.value
                contador += 1
                heapq.heappush(frontera, (hijo.heuristica, contador, hijo))
    return [], orden


def a_estrella(inicio, objetivo, verbose=False):
    reiniciar()
    contador = 0
    g = {inicio.value: 0}
    padres = {inicio.value: None}
    frontera = [(inicio.heuristica, contador, inicio)]
    orden = []
    while frontera:
        f, _, actual = heapq.heappop(frontera)
        if actual.visitado:
            continue
        actual.visitado = True
        orden.append(actual.value)
        if verbose:
            print(len(orden), "Visitando nodo:", actual.value,
                  "| g =", round(g[actual.value], 1), "| f =", round(f, 1))
        if actual == objetivo:
            return reconstruir(padres, actual.value), orden
        for hijo, costo in actual.hijos:
            nuevo_g = g[actual.value] + costo
            if hijo.value not in g or nuevo_g < g[hijo.value]:
                g[hijo.value] = nuevo_g
                padres[hijo.value] = actual.value
                contador += 1
                heapq.heappush(frontera, (nuevo_g + hijo.heuristica, contador, hijo))
    return [], orden


ALGORITMOS = {"BFS": bfs, "DFS": dfs, "Greedy Best-First": greedy, "A*": a_estrella}

EXPLICACIONES = {
    "BFS": ("Explora por niveles (primero vecinos a 1 conexión, luego a 2, etc.). Garantiza "
            "el MENOR NÚMERO DE CONEXIONES, pero ignora los costos."),
    "DFS": ("Baja por una rama hasta el fondo y, si no llega, retrocede (backtracking). "
            "Encuentra UNA ruta, no necesariamente corta ni barata; depende del orden de los vecinos."),
    "Greedy Best-First": ("Expande siempre el nodo con menor heurística h(n). Es rápido, pero "
                          "ignora el costo acumulado y puede caer en rutas caras."),
    "A*": ("Expande el nodo con menor f(n) = g(n) + h(n) (costo acumulado + estimación). Con "
           "heurística admisible garantiza la ruta de MENOR COSTO."),
}


def ejecutar(nombre, inicio, objetivo, heuristica=h_euclidiana, repeticiones=200):
    calcular_heuristicas(objetivo, heuristica)
    funcion = ALGORITMOS[nombre]
    ruta, orden = funcion(inicio, objetivo)
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        funcion(inicio, objetivo)
    tiempo_ms = (time.perf_counter() - t0) / repeticiones * 1000
    return {
        "algoritmo": nombre, "ruta": ruta, "orden": orden,
        "explorados": len(orden), "tiempo_ms": tiempo_ms,
        "costo": costo_ruta(ruta) if ruta else None,
        "conexiones": len(ruta) - 1 if ruta else None,
        "explicacion": EXPLICACIONES[nombre],
    }


def texto_resultado(r):
    if not r["ruta"]:
        return f"[{r['algoritmo']}] No existe ruta disponible.\n"
    return (
        f"=== {r['algoritmo']} ===\n"
        f"Ruta encontrada : {' -> '.join(r['ruta'])}\n"
        f"Conexiones      : {r['conexiones']}\n"
        f"Costo total     : {r['costo']}\n"
        f"Nodos explorados: {r['explorados']}  ({' '.join(r['orden'])})\n"
        f"Tiempo ejecución: {r['tiempo_ms']:.4f} ms (promedio de 200 corridas)\n"
        f"Explicación     : {r['explicacion']}\n"
    )


def modo_consola():
    ini, meta = NODOS[INICIO_DEFECTO], NODOS[DESTINO_DEFECTO]
    print(f"Inicio: {ini.value} ({ini.nombre})   Destino: {meta.value} ({meta.nombre})\n")
    resultados = []
    for nombre in ALGORITMOS:
        calcular_heuristicas(meta)
        print(f"--- {nombre} ---")
        ALGORITMOS[nombre](ini, meta, verbose=True)     # imprime "Visitando nodo"
        r = ejecutar(nombre, ini, meta)
        resultados.append(r)
        print(texto_resultado(r))
    print("TABLA COMPARATIVA")
    print(f"{'Algoritmo':<20}{'Ruta':<26}{'Costo':<8}{'Explorados':<11}{'Tiempo (ms)'}")
    for r in resultados:
        print(f"{r['algoritmo']:<20}{'-'.join(r['ruta']):<26}{r['costo']:<8}"
              f"{r['explorados']:<11}{r['tiempo_ms']:.4f}")
    print("\nEXPERIMENTO: cambiar la heurística")
    print(f"{'Heurística':<32}{'Alg.':<8}{'Ruta':<24}{'Costo':<8}{'Explorados'}")
    for hn, hf in HEURISTICAS.items():
        for alg in ("Greedy Best-First", "A*"):
            r = ejecutar(alg, ini, meta, hf, repeticiones=1)
            print(f"{hn:<32}{alg[:6]:<8}{'-'.join(r['ruta']):<24}{r['costo']:<8}{r['explorados']}")


# ----------------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA (Tkinter)
# ----------------------------------------------------------------------------
def modo_grafico():
    import tkinter as tk
    from tkinter import ttk

    ANCHO, ALTO = 780, 540

    def pos(n):
        return 50 + n.x * 7.2, ALTO - (40 + n.y * 5.4)

    root = tk.Tk()
    root.title("Laboratorio de algoritmos de búsqueda - Logística")
    panel = ttk.Frame(root, padding=8)
    panel.grid(row=0, column=0, sticky="ns")
    canvas = tk.Canvas(root, width=ANCHO, height=ALTO, bg="white")
    canvas.grid(row=0, column=1, padx=6, pady=6)

    opciones = [f"{n.value} - {n.nombre}" for n in NODOS.values()]
    var_alg = tk.StringVar(value="A*")
    var_ini = tk.StringVar(value=opciones[0])
    var_fin = tk.StringVar(value=opciones[list(NODOS).index(DESTINO_DEFECTO)])
    var_h = tk.StringVar(value=list(HEURISTICAS)[0])

    for etiqueta, var, valores in (("Algoritmo", var_alg, list(ALGORITMOS)),
                                   ("Nodo inicial", var_ini, opciones),
                                   ("Nodo destino", var_fin, opciones),
                                   ("Heurística (Greedy / A*)", var_h, list(HEURISTICAS))):
        ttk.Label(panel, text=etiqueta).pack(anchor="w")
        ttk.Combobox(panel, textvariable=var, values=valores,
                     state="readonly", width=28).pack(pady=2)

    salida = tk.Text(panel, width=44, height=24, wrap="word", font=("Consolas", 9))

    def dibujar(ruta=None, explorados=None, ini=None, fin=None):
        canvas.delete("all")
        aristas = set()
        if ruta:
            for i in range(len(ruta) - 1):
                aristas.add((ruta[i], ruta[i + 1]))
                aristas.add((ruta[i + 1], ruta[i]))
        for a, b, costo, bloq in DATOS_CONEXIONES:
            x1, y1 = pos(NODOS[a])
            x2, y2 = pos(NODOS[b])
            if bloq:
                canvas.create_line(x1, y1, x2, y2, fill="red", dash=(5, 4), width=2)
                texto, color = f"{costo} X", "red"
            elif (a, b) in aristas:
                canvas.create_line(x1, y1, x2, y2, fill="#1b9e3e", width=5)
                texto, color = str(costo), "#1b9e3e"
            else:
                canvas.create_line(x1, y1, x2, y2, fill="#999999", width=2)
                texto, color = str(costo), "#444444"
            canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2 - 7, text=texto,
                               fill=color, font=("Arial", 8, "bold"))
        for v, n in NODOS.items():
            x, y = pos(n)
            relleno = "#dddddd"
            if explorados and v in explorados:
                relleno = "#bfdcff"
            if ruta and v in ruta:
                relleno = "#9be7a8"
            if v == ini:
                relleno = "#2ecc71"
            if v == fin:
                relleno = "#ff6b6b"
            canvas.create_oval(x - 17, y - 17, x + 17, y + 17, fill=relleno, outline="black", width=2)
            canvas.create_text(x, y, text=v, font=("Arial", 11, "bold"))
            canvas.create_text(x, y + 28, text=n.nombre, font=("Arial", 7))
        canvas.create_text(10, 10, anchor="nw", font=("Arial", 8),
                           text="Verde=inicio  Rojo=destino  Azul=explorado  Línea verde=ruta  "
                                "Roja punteada=bloqueada")

    def elegir():
        return NODOS[var_ini.get().split(" - ")[0]], NODOS[var_fin.get().split(" - ")[0]]

    def buscar():
        ini, fin = elegir()
        r = ejecutar(var_alg.get(), ini, fin, HEURISTICAS[var_h.get()])
        salida.delete("1.0", "end")
        if var_alg.get() in ("Greedy Best-First", "A*"):
            salida.insert("end", f"Heurística: {var_h.get()}\n\n")
        salida.insert("end", texto_resultado(r))
        dibujar(r["ruta"], set(r["orden"]), ini.value, fin.value)

    def comparar():
        ini, fin = elegir()
        salida.delete("1.0", "end")
        salida.insert("end", f"{'Alg.':<8}{'Costo':<7}{'Expl.':<7}Ruta\n")
        for nombre in ALGORITMOS:
            r = ejecutar(nombre, ini, fin, HEURISTICAS[var_h.get()], repeticiones=50)
            if r["ruta"]:
                salida.insert("end", f"{nombre[:6]:<8}{r['costo']:<7}{r['explorados']:<7}"
                                     f"{'-'.join(r['ruta'])}\n")
            else:
                salida.insert("end", f"{nombre[:6]:<8}sin ruta\n")
        dibujar(None, None, ini.value, fin.value)

    ttk.Button(panel, text="Ejecutar búsqueda", command=buscar).pack(fill="x", pady=(8, 2))
    ttk.Button(panel, text="Comparar los 4 algoritmos", command=comparar).pack(fill="x", pady=2)
    salida.pack(pady=6)
    dibujar(ini=INICIO_DEFECTO, fin=DESTINO_DEFECTO)
    root.mainloop()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        modo_consola()
    else:
        modo_grafico()