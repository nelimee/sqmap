import typing as ty

import numpy
import matplotlib.pyplot as plt

from sqmap.backends import QubitPlacement, get_qubit_placement
from sqmap.visualisation.flatmap_cartopy import plot_over_projected_bloch_sphere_2d


def plot_whole_chip_view(
    density_matrices: ty.List[ty.List[ty.Tuple[numpy.ndarray, numpy.ndarray]]],
    processor_type: str,
):
    """Plot generic information about the whole chip.

    :param density_matrices: a list containing, for each qubits, a list of tuples
        composed of the ideal point (in cartesian coordinate) on the Bloch sphere
        and the computed density matrix.
    :param processor_type: type of the processor used for the experiment.
    """
    placement: QubitPlacement = get_qubit_placement(processor_type)
    if len(density_matrices) != placement.qubit_number:
        raise RuntimeError(
            f"Different number of qubits between processor type ({placement.qubit_number})"
            f" and provided data ({len(density_matrices)})"
        )

    fig_size = (placement.max_y + 1, placement.max_x + 1)
    fig, axes = plt.subplots(*fig_size)
    fig.set_tight_layout(True)
    for qubit_index, (x, y) in enumerate(placement.positions):
        ideal_points = [dm[0] for dm in density_matrices[qubit_index]]
        density_matrices_on_points = [dm[1] for dm in density_matrices[qubit_index]]
        plot_over_projected_bloch_sphere_2d(
            ideal_points,
            density_matrices_on_points,
            fig=fig,
            ax=axes[y, x],
            figure_subplots_indices=fig_size,
            ax_index=y * fig_size[1] + x + 1,
        )
    all_axes: ty.Set[ty.Tuple[int, int]] = {
        (i, j) for i in range(placement.max_x + 1) for j in range(placement.max_y + 1)
    }
    for x, y in all_axes.difference(placement.positions):
        axes[y, x].remove()
