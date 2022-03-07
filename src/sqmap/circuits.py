import typing as ty

from qiskit import QuantumCircuit

from sqt.basis.equidistant import get_equidistant_points, point_to_circuit


def get_approximately_equidistant_circuits(
    approximate_point_number: int,
) -> ty.List[QuantumCircuit]:
    """Construct and returns circuits that are approximately equidistant.

    This function will construct approximately approximate_point_number
    quantum circuits that will prepare quantum states that are approximately
    equidistant when placed on the Bloch sphere.

    :param approximate_point_number: number of circuits that will be generated.
        This is only an approximation, the actual number of circuits that will
        be generated might be slightly lower or higher.
    :return: the generated circuits. Each circuit will have a name that is
        the list representation of the point it prepares. This point can
        be retrieved with the following code (yes it is ugly, I am searching
        for a better alternative):
            circuits = get_approximately_equidistant_circuits(10)
            points = [eval(c.name) for c in circuits]
    """
    return [
        point_to_circuit(point, f"{point.tolist()}")
        for point in get_equidistant_points(approximate_point_number)
    ]
