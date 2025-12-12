from tsp import TSP
from pathlib import Path
import random
import csv
import sys

# Import solvers
from gg_gurobi import gg_gurobi_solve
from gg_cplex import gg_cplex_solve
from mtz_gurobi import mtz_gurobi_solve
from mtz_cplex import mtz_cplex_solve
from utils import instance_loader

CURRENT_DIR = Path.cwd()
DIR_INSTANCES = CURRENT_DIR / "instances"
DIR_INSTANCES_S = DIR_INSTANCES / "small"
DIR_INSTANCES_M = DIR_INSTANCES / "medium"
DIR_INSTANCES_L = DIR_INSTANCES / "large"

# Instance definitions
small_instances = ["br17.atsp", "ftv33.atsp", "p43.atsp", "ry48p.atsp"]
# small_instances = ["p43.atsp"] # Commented out to run full benchmark
medium_instances = ["ft70.atsp", "ftv170.atsp", "ftv64.atsp", "kro124p.atsp"]
large_instances = ["rbg323.atsp", "rbg358.atsp"]

VISUALIZE = False

def test(out_dir):
    """
    Ejecuta el benchmark completo:
    1. Carga todas las instancias.
    2. Ejecuta los 4 solvers (GG/MTZ x CPLEX/Gurobi) para cada instancia.
    3. Guarda resultados en CSV incrementalmente.
    """
    # Preparar directorio y archivo de salida
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "resultados.csv"
    
    # Encabezados 
    fieldnames = [
        "instancia", "num_nodos", "modelo", "solver", 
        "num_vars", "numrest", "tiempo(s)", "por_gap", "func_obj"
    ]

    # Crear archivo y escribir header si no existe
    if not csv_file.exists():
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
    print(f"--- Iniciando Benchmark ---")
    print(f"Guardando resultados en: {csv_file}")

    # Cargar todas las instancias
    problem_dict = instance_loader(
        small_instances,
        medium_instances,
        large_instances,
        DIR_INSTANCES_S,
        DIR_INSTANCES_M,
        DIR_INSTANCES_L,
    )
    
    # Configuración
    TIME_LIMIT = 3600  # 1 hora
    solvers = [
        gg_cplex_solve,
        gg_gurobi_solve,
        mtz_cplex_solve,
        mtz_gurobi_solve
    ]
    
    categories = ["small", "medium", "large"]

    for category in categories:
        instances = problem_dict.get(category, [])
        for problem in instances:
            for solve_func in solvers:
                try:
                    # Ejecutar el solver
                    # Retorna (dict_resultados, matriz_solucion)
                    res_dict, _ = solve_func(problem, time_limit=TIME_LIMIT)
                    
                    # Mapear claves del diccionario interno al formato CSV 
                    row = {
                        "instancia": res_dict.get("instancia"),
                        "num_nodos": res_dict.get("num_nodos"),
                        "modelo": res_dict.get("modelo"),
                        "solver": res_dict.get("solver"),
                        "num_vars": res_dict.get("num_vars"),
                        "numrest": res_dict.get("num_rest"),       # Mapping
                        "tiempo(s)": res_dict.get("tiempo_(s)"),   # Mapping
                        "por_gap": res_dict.get("por_gap"),
                        "func_obj": res_dict.get("func_obj"),
                    }

                    # Escribir inmediatamente al archivo (append mode)
                    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(row)
                        
                except Exception as e:
                    print(f"!! Error resolviendo {problem.name} con {solve_func.__name__}: {e}")
                    # Opcional: Escribir fila de error en CSV para registro
                    continue

    print("--- Benchmark Finalizado ---")


if __name__ == "__main__":
    # Ejecuta el benchmark y guarda en la carpeta 'resultados'
    test("resultados")
