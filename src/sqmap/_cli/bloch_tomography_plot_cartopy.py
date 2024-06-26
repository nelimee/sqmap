import argparse
import pickle
import typing as ty
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from sqmap.visualisation.flatmap_cartopy import plot_over_projected_bloch_sphere_2d


def get_data(
    post_processed_data_backup: Path,
) -> ty.Tuple[
    int, ty.List[ty.List[ty.Tuple[numpy.ndarray, numpy.ndarray]]], str, str, str
]:
    with open(post_processed_data_backup, "rb") as f:
        data = pickle.load(f)

    return (
        data["qubit_number"],
        data["density_matrices"],
        data["backend_name"],
        data["basis_name"],
        data["post_processing_method"],
    )


# def compute_z_data(ideal: numpy.ndarray, obtained: numpy.ndarray) -> float:
#     from qiskit.opflow import Zero, One
#     from qiskit.quantum_info.states import DensityMatrix, state_fidelity

#     state = numpy.ones((2,)) / numpy.sqrt(2)
#     rho = numpy.outer(state, state)

#     return 1 - state_fidelity(DensityMatrix(rho), DensityMatrix(ideal), validate=False)


def main():
    parser = argparse.ArgumentParser(
        description=("Plot the reconstructed density matrices.")
    )
    parser.add_argument(
        "backup_file",
        type=Path,
        help="Backup file path that has been saved after computing the density matrices.",
    )
    parser.add_argument(
        "-i",
        "--qubit-index",
        type=int,
        default=None,
        help="Index of the qubit to plot. Default to all qubits if not given.",
    )
    args = parser.parse_args()

    (
        qubit_number,
        density_matrices,
        backend_name,
        basis_name,
        post_processing_method,
    ) = get_data(args.backup_file)

    qubit_indices: ty.List[int]
    if args.qubit_index is None:
        qubit_indices = list(range(qubit_number))
    else:
        qubit_indices = [args.qubit_index]

    for qubit_index in qubit_indices:
        qubit_density_matrices = density_matrices[qubit_index]
        ideal_cartesian_points = [point for point, _ in qubit_density_matrices]
        obtained_density_matrices = [
            density_matrix for _, density_matrix in qubit_density_matrices
        ]
        plot_over_projected_bloch_sphere_2d(
            ideal_cartesian_points,
            obtained_density_matrices,
            title=(
                f"Qubit n°{qubit_index} of backend '{backend_name}' using '{basis_name}'"
                f" tomography basis and '{post_processing_method}' reconstruction method."
            ),
        )
    plt.show()
