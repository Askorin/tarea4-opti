# mtz_gurobi.py
from gurobipy import *
import networkx as nx
import numpy as np


def make_mtz_gurobi_model(problem):
    """
    Construye un modelo Gurobi para ATSP usando la formulación MTZ.
    Retorna: modelo, x (vars binarias)
    """

    mdl = Model(f"mtz_{problem.name}")

    G = problem.G
    n = problem.n
    c = nx.adjacency_matrix(G)

    # --- VARIABLES ---
    x = mdl.addVars(G.edges(), vtype=GRB.BINARY, name="x")

    # variables u_i para MTZ
    # u[0] = 0, u[i] ∈ [0, n-1]
    u = mdl.addVars(G.nodes(), vtype=GRB.CONTINUOUS, lb=0.0, ub=n - 1, name="u")

    mdl.update()

    # fijar raíz
    mdl.addConstr(u[0] == 0, "fix_root")

    # --- RESTRICCIONES DE GRADO ---
    for i in G.nodes():
        mdl.addConstr(x.sum(i, "*") == 1, name=f"out_{i}")
        mdl.addConstr(x.sum("*", i) == 1, name=f"in_{i}")

    # --- RESTRICCIONES MTZ ---
    M = n - 1
    for i, j in G.edges():
        if i == 0 or j == 0:
            continue
        mdl.addConstr(u[i] - u[j] + M * x[i, j] <= n - 2,
                      name=f"mtz_{i}_{j}")

    # --- OBJETIVO ---
    mdl.setObjective(quicksum(c[i, j] * x[i, j] for i, j in G.edges()),
                     GRB.MINIMIZE)

    return mdl, x



def mtz_gurobi_solve(problem, time_limit: int):
    """
    Resuelve ATSP usando MTZ + Gurobi.
    Retorna:
      1. diccionario con resultados
      2. matriz x[i][j]
    """

    mdl, x = make_mtz_gurobi_model(problem)

    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("OutputFlag", 0)
    mdl.optimize()

    # --- Datos base ---
    instance = problem.name
    n = problem.n
    model_name = "mtz"
    solver = "gurobi"

    num_vars = mdl.NumVars
    num_constrs = mdl.NumConstrs
    cpu_time = mdl.Runtime

    gap_str = "N/A"
    func = "N/A"

    print(f"Resolviendo {instance}")

    if mdl.SolCount > 0:
        gap = mdl.MIPGap
        func = mdl.ObjVal

        if mdl.status == GRB.OPTIMAL:
            gap_str = "0.00%"
        else:
            gap_str = f"{gap*100:.6f}%"

    print(f"Resultado de {instance}: F.O = {func}, Gap = {gap_str}, Tiempo = {cpu_time:.2f}s")

    # --- Diccionario salida ---
    solution_dict = {
        "instancia": instance,
        "num_nodos": n,
        "modelo": model_name,
        "solver": solver,
        "num_vars": num_vars,
        "num_rest": num_constrs,
        "tiempo_(s)": cpu_time,
        "por_gap": gap_str,
        "func_obj": func,
    }

    # --- Matriz solución ---
    x_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            try:
                x_matrix[i, j] = x[i, j].X
            except:
                x_matrix[i, j] = 0

    return solution_dict, x_matrix
