from tsp import TSP
from pathlib import Path
import random

CURRENT_DIR = Path.cwd()
DIR_INSTANCES = CURRENT_DIR / "instances"
DIR_INSTANCES_S = DIR_INSTANCES / "small"
DIR_INSTANCES_M = DIR_INSTANCES / "medium"
DIR_INSTANCES_L = DIR_INSTANCES / "large"


# 4-4-2 para un total de 10
small_instances = ["br17.atsp", "ftv33.atsp", "p43.atsp", "ry48p.atsp"]
medium_instances = ["ft70.atsp", "ftv170.atsp", "ftv64.atsp", "kro124p.atsp"]
large_instances = ["rbg323.atsp", "rbg358.atsp"]


def instance_loader() -> dict:
    """
    Retorna un diccionario con los problemas ATSP en la carpeta de instancias.
    """
    problem_dict = {
        "small": [],
        "medium": [],
        "large": []
    }
    
    for instance in small_instances:
        problem_dict["small"].append(
                TSP(DIR_INSTANCES_S / instance, name=instance.replace(".atsp", ""))
        )

    for instance in medium_instances:
        problem_dict["medium"].append(
                TSP(DIR_INSTANCES_M / instance, name=instance.replace(".atsp", ""))
        )

    for instance in large_instances:
        problem_dict["large"].append(
                TSP(DIR_INSTANCES_L / instance, name=instance.replace(".atsp", ""))
        )

    return problem_dict

if __name__ == "__main__":
    # Ejemplo de uso

    problem_dict = instance_loader()

    # Un problema pequeño al azar
    problemita = random.choice(problem_dict["small"])

    # Solución dummy
    dummy_sol = range(problemita.n)

    # Calculamos su costo
    print(f"Costo: {problemita.evaluate_solution(dummy_sol)}")

    # Visualizamos
    problemita.visualize(dummy_sol, title=f"Visualización {problemita.name}")

