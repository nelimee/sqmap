import argparse
import typing as ty
from pathlib import Path
import pickle

import numpy
import matplotlib.pyplot as plt

from sqmap.visualisation.flatmap import plot_bloch_vector_displacement_arrow_field_2d


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
        plot_bloch_vector_displacement_arrow_field_2d(
            ideal_cartesian_points,
            obtained_density_matrices,
            title=(
                f"Qubit n°{qubit_index} of backend '{backend_name}' using '{basis_name}'"
                f" tomography basis and '{post_processing_method}' reconstruction method."
            ),
        )
    plt.show()
