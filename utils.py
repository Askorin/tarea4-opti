from tsp import TSP


def instance_loader(
    small_instances,
    medium_instances,
    large_instances,
    dir_instances_s,
    dir_instances_m,
    dir_instances_l,
) -> dict:
    """
    Retorna un diccionario con los problemas ATSP en la carpeta de instancias.
    """
    problem_dict = {"small": [], "medium": [], "large": []}

    for instance in small_instances:
        problem_dict["small"].append(
            TSP(dir_instances_s / instance, name=instance.replace(".atsp", ""))
        )

    for instance in medium_instances:
        problem_dict["medium"].append(
            TSP(dir_instances_m / instance, name=instance.replace(".atsp", ""))
        )

    for instance in large_instances:
        problem_dict["large"].append(
            TSP(dir_instances_l / instance, name=instance.replace(".atsp", ""))
        )

    return problem_dict
