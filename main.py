from tsp import TSP
from pathlib import Path
import random
from gg_gurobi import gg_gurobi_solve, make_gg_gurobi_model
from utils import instance_loader

CURRENT_DIR = Path.cwd()
DIR_INSTANCES = CURRENT_DIR / "instances"
DIR_INSTANCES_S = DIR_INSTANCES / "small"
DIR_INSTANCES_M = DIR_INSTANCES / "medium"
DIR_INSTANCES_L = DIR_INSTANCES / "large"


# 4-4-2 para un total de 10
small_instances = ["br17.atsp", "ftv33.atsp", "p43.atsp", "ry48p.atsp"]
medium_instances = ["ft70.atsp", "ftv170.atsp", "ftv64.atsp", "kro124p.atsp"]
large_instances = ["rbg323.atsp", "rbg358.atsp"]

VISUALIZE = False

if __name__ == "__main__":

    # Ejemplo de uso

    problem_dict = instance_loader(
        small_instances,
        medium_instances,
        large_instances,
        DIR_INSTANCES_S,
        DIR_INSTANCES_M,
        DIR_INSTANCES_L,
    )

    # Un problema pequeño al azar
    problemita = random.choice(problem_dict["small"])

    # Solución dummy
    dummy_sol = range(problemita.n)

    # Calculamos su costo
    print(f"Costo: {problemita.evaluate_solution(dummy_sol)}")

    # Visualizamos
    problemita.visualize(dummy_sol, title=f"Visualización {problemita.name}")

    # --- Ejemplo de Resoluciones Problemas Pequeños

    for problem in problem_dict["small"]:
        print(
            f"Intentando resolver {problem.name} con formulación gg, solver gurobi...\n"
        )
        data, solution_matrix = gg_gurobi_solve(problem, time_limit=3600)
        tour = problem.validate_solution_matrix(solution_matrix)
        if tour:
            print(f"Solución es válida{'' if not VISUALIZE else ', visualizando...'}")
            print(f"Costo verificado: {problem.evaluate_solution(tour)}")
            if VISUALIZE:
                problem.visualize(tour, title=f"Visualización {problem.name}")

        print("\n----------------------------\n")
