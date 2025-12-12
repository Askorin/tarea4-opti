from tsp import TSP
from gurobipy import *
import networkx as nx
import numpy as np

def make_gg_gurobi_model(problem: TSP) -> tuple:
    """
    Construye y retorna un modelo de Gurobi a partir del problema ATSP, con formulación GG
    Adicionalmente retorna la variable de decisión
    """

    mdl = Model(f"gg_{problem.name}") # Fixed string formatting

    G = problem.G 
    n = problem.n 
    c = nx.adjacency_matrix(G) 

    # --- VARIABLES DE DECISIÓN ---
    # addVars devuelve un 'tupledict' que permite usar métodos .sum()
    x = mdl.addVars(G.edges(), vtype=GRB.BINARY, name="x")
    y = mdl.addVars(G.edges(), vtype=GRB.CONTINUOUS, name="y", lb=0.0)

    # --- RESTRICCIONES ---

    # 1. Grado: Cada nodo tiene un arco de entrada y uno de salida
    for i in G.nodes():
        # Usamos x.sum(i, '*') para evitar KeyError si el grafo no es completo
        mdl.addConstr(x.sum(i, '*') == 1, name=f"out_degree_{i}")
        mdl.addConstr(x.sum('*', i) == 1, name=f"in_degree_{i}")

    # 2. Fuente (flujo): El nodo 0 envía n - 1 unidades en total
    mdl.addConstr(y.sum(0, '*') == n - 1, name="source_flow")

    # 3. Conservación de flujo 
    for i in G.nodes():
        if i == 0:
            continue
        
        # y.sum('*', i) es la suma de todo lo que entra a i
        # y.sum(i, '*') es la suma de todo lo que sale de i
        mdl.addConstr(
            y.sum('*', i) - y.sum(i, '*') == 1, 
            name=f"flow_bal_{i}"
        )

    # 4. Capacidad 
    for i, j in G.edges():
        mdl.addConstr(y[i, j] <= (n - 1) * x[i, j], name=f"cap_{i}_{j}")
    
    # --- FO ---
    mdl.setObjective(quicksum(c[i, j] * x[i, j] for i, j in G.edges()), GRB.MINIMIZE)
    
    return mdl, x

def gg_gurobi_solve(problem: TSP, time_limit: int) -> tuple[dict, np.ndarray]:
    """
    Resuelve un problema de ATSP con la formulacion GG, utilizando gurobi.
    Retorna una tupla:
        1. Un diccionario con los datos de la solución, tiempo de ejecución, metadata, etc.
        2. La matriz de decisión
    """

    mdl, x = make_gg_gurobi_model(problem)
    print(f"Resolviendo {problem.name}")
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("TimeLimit", time_limit)
    mdl.optimize()

    # Datos para el CSV
    instance = problem.name
    num_nodes = problem.n
    model = "gg"
    solver ="gurobi"
    num_vars = mdl.Numvars
    num_constrs = mdl.NumConstrs
    cpu_time = mdl.Runtime
    gap_str = "N/A"
    func = "N/A"

    if mdl.SolCount > 0:
        gap = mdl.MIPGap
        func = mdl.ObjVal

        if mdl.status == GRB.OPTIMAL:
            gap_str = "0.00%"
        else:
            gap_str = f"{gap * 100:.6f}%"


        x_solution_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                x_solution_matrix[i, j] = x[i, j].X

    elif model.status == GRB.INFEASIBLE:
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
        "func_obj": func
    }



    return solution_dict, x_solution_matrix
