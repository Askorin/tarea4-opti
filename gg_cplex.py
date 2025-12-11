from docplex.mp.model import Model
from tsp import TSP
import networkx as nx

def make_gg_cplex_model(problem: TSP):
    mdl = Model(name = f"gg_cplex_{problem.name}")
    
    # --------------- Parametros ---------------
    G = problem.G                               # Grafo del problema
    n = problem.n                               # Numero de nodos
    c = nx.to_numpy_array(G, weight="weight")   # matriz de costos 

    # ---------------- Variables ----------------
    x = mdl.binary_var_dict(G.edges(), name="x")
    y = mdl.continuous_var_dict(G.edges(), lb=0.0, name="y")

    # -------------- Restricciones --------------
    
    # 1. Grado: Cada nodo tiene un arco de entrada y uno de salida
    for i in G.nodes():
        # (Restricción 2b)
        mdl.add_constraint(
            mdl.sum(x[i, j] for j in (G.successors(i) if G.is_directed() else G.neighbors(i))) == 1,
            ctname=f"out_degree_{i}"
        )
        # (Restricción 2c)
        mdl.add_constraint(
            mdl.sum(x[j, i] for j in (G.predecessors(i) if G.is_directed() else G.neighbors(i))) == 1,
            ctname=f"in_degree_{i}"
        )
    
    # 2. Fuente (flujo): El nodo 0 envía n - 1 unidades en total
    # (Restricción 2d)
    mdl.add_constraint(
        mdl.sum(y[0, j] for j in G.neighbors(0)) == n - 1,
        ctname="source_flow"
    )

    # 3. Conservación de flujo
    for i in G.nodes():
        if i == 0:
            continue
        
        in_nei  = (G.predecessors(i) if G.is_directed() else G.neighbors(i))
        out_nei = (G.successors(i)  if G.is_directed() else G.neighbors(i))

        # (Restricción 2e)
        mdl.add_constraint(
            mdl.sum(y[j, i] for j in in_nei) - mdl.sum(y[i, j] for j in out_nei) == 1,
            ctname=f"flow_bal_{i}"
        )

    # 4. Capacidad 
    # (Restricción 2f)
    for i, j in G.edges():
        mdl.add_constraint(
            y[i, j] <= (n - 1) * x[i, j],
            ctname=f"cap_{i}_{j}"
        )

    # -------------- Función Objetivo --------------
    mdl.minimize(
        mdl.sum(c[i, j] * x[i, j] for i, j in G.edges())
    )

    return mdl

def gg_cplex_solve(problem: TSP, time_limit: int) -> dict:
    mdl = make_gg_cplex_model(problem)
    print(f"Resolviendo {problem.name}")

    mdl.set_time_limit(time_limit)
    mdl.parameters.mip.display = 0 
    
    sol = mdl.solve(log_output=False)
    status = mdl.solve_details.status

    # Datos para el CSV
    instance = problem.name
    num_nodes = problem.n
    model = "gg"
    solver = "cplex"
    num_vars = mdl.number_of_variables
    num_constrs = mdl.number_of_constraints
    cpu_time = mdl.solve_details.time
    gap_str = "N/A"
    func = "N/A"

    if sol is not None:
        func = mdl.objective_value
        gap = mdl.solve_details.mip_relative_gap

        if status == "optimal":
            gap_str = "0.00%"
        else:
            gap_str = f"{gap * 100:.6f}%"
    else:
        func = "INFACTIBLE"

    print(
        f"Resultado de {instance}: F.O = {func}, Gap = {gap_str}, Tiempo = {cpu_time:.2f}s"
    )

    solution_dict = {
        "instancia": instance,
        "num_nodos": num_nodes,
        "modelo": model,
        "solver": solver,
        "num_vars": num_vars,
        "num_rest": num_constrs,
        "tiempo_(s)": cpu_time,
        "por_gap": gap_str,
        "func_obj": func,
    }

    return solution_dict

