from docplex.mp.model import Model
from tsp import TSP
import networkx as nx
import numpy as np

def make_mtz_cplex_model(problem: TSP):
    mdl = Model(name=f"mtz_cplex_{problem.name}")

    # --------------- Parametros ---------------
    G = problem.G
    n = problem.n
    c = nx.to_numpy_array(G, weight="weight")

    # ---------------- Variables ----------------
    # Variables binarias x_ij para los arcos
    x = mdl.binary_var_dict(G.edges(), name="x")
    
    # Variables continuas u_i para el orden de visita (MTZ)
    # Generalmente u_i varía entre 1 y n (o 0 y n-1). 
    # Aquí seguimos la lógica del snippet original: lb=1, ub=n
    # Solo necesitamos u para i != 0, pero definirlos para todos simplifica la indexación.
    u = mdl.continuous_var_dict(G.nodes(), lb=1, ub=n, name="u")

    # -------------- Restricciones --------------

    # 1. Grado: Cada nodo tiene un arco de entrada y uno de salida
    for i in G.nodes():
        # Arcos salientes
        out_neighbors = list(G.successors(i)) if G.is_directed() else list(G.neighbors(i))
        mdl.add_constraint(
            mdl.sum(x[i, j] for j in out_neighbors) == 1,
            ctname=f"out_degree_{i}"
        )
        # Arcos entrantes
        in_neighbors = list(G.predecessors(i)) if G.is_directed() else list(G.neighbors(i))
        mdl.add_constraint(
            mdl.sum(x[j, i] for j in in_neighbors) == 1,
            ctname=f"in_degree_{i}"
        )

    # 2. MTZ - Eliminación de Subtours
    # Fijamos u[0] = 1 para romper simetría
    mdl.add_constraint(u[0] == 1, ctname="fix_u0")

    # Restricción MTZ estándar: u_i - u_j + n * x_ij <= n - 1
    # Aplica para todo i, j != 0
    for i, j in G.edges():
        if i != 0 and j != 0:
            mdl.add_constraint(
                u[i] - u[j] + n * x[i, j] <= n - 1,
                ctname=f"mtz_{i}_{j}"
            )

    # -------------- Función Objetivo --------------
    mdl.minimize(
        mdl.sum(c[i, j] * x[i, j] for i, j in G.edges())
    )

    return mdl, x

def mtz_cplex_solve(problem: TSP, time_limit: int) -> tuple[dict, np.ndarray]:
    mdl, x = make_mtz_cplex_model(problem)
    print(f"Resolviendo {problem.name} (MTZ - CPLEX)...")

    mdl.set_time_limit(time_limit)
    mdl.parameters.mip.display = 0

    sol = mdl.solve(log_output=False)
    
    # Metadata básica
    instance = problem.name
    num_nodes = problem.n
    model_name = "mtz"
    solver_name = "cplex"
    num_vars = mdl.number_of_variables
    num_constrs = mdl.number_of_constraints
    
    # Inicializar valores por defecto
    cpu_time = 0
    func = "INFACTIBLE"
    gap_str = "N/A"
    
    # Matriz vacía
    x_solution_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    if sol is not None:
        cpu_time = mdl.solve_details.time
        func = mdl.objective_value
        gap = mdl.solve_details.mip_relative_gap
        status = mdl.solve_details.status
        
        # Formato de gap
        gap_str = f"{gap * 100:.6f}%"

        print(f"Resultado de {instance}: F.O = {func}, Gap = {gap_str}, Tiempo = {cpu_time:.2f}s, Status: {status}")

        # --- Extracción de Solución (Igual que GG) ---
        for (u, v), var in x.items():
            if round(var.solution_value) == 1:
                x_solution_matrix[u, v] = 1
    else:
        print(f"No se encontró solución para {instance}")

    solution_dict = {
        "instancia": instance,
        "num_nodos": num_nodes,
        "modelo": model_name,
        "solver": solver_name,
        "num_vars": num_vars,
        "num_rest": num_constrs,
        "tiempo_(s)": cpu_time,
        "por_gap": gap_str,
        "func_obj": func,
    }

    return solution_dict, x_solution_matrix
