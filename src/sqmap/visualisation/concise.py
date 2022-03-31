import typing as ty

import numpy
import matplotlib.pyplot as plt

from sqmap.visualisation.flatmap import account_for_periodicity, _compute_infidelity
from sqmap.visualisation.transformers import (
    cartesian2density,
    density2spherical,
    cartesian2spherical,
    spherical2geographic,
)


def plot_bloch_concise_vector_displacement_arrow_field_2d(
    ideal_points_cartesian: ty.List[numpy.ndarray],
    density_matrices: ty.List[numpy.ndarray],
    compute_z_data: ty.Callable[
        [numpy.ndarray, numpy.ndarray], float
    ] = _compute_infidelity,
    fig=None,
    ax=None,
    cax=None,
):
    """"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Compute the displacement vectors
    obtained_points_spherical = numpy.array(
        [density2spherical(rho) for rho in density_matrices]
    )
    ideal_points_spherical = numpy.array(
        [cartesian2spherical(*p) for p in ideal_points_cartesian]
    )
    ideal_points_density_matrices = numpy.array(
        [cartesian2density(*p) for p in ideal_points_cartesian]
    )
    # Compute the longitude and latitude.
    lat_ideal, lon_ideal = spherical2geographic(
        ideal_points_spherical[:, 1], ideal_points_spherical[:, 2]
    )
    lat_obtained, lon_obtained = spherical2geographic(
        obtained_points_spherical[:, 1], obtained_points_spherical[:, 2]
    )
    # Compute the values that will be displayed as a heat map.
    Z = numpy.array(
        [
            compute_z_data(ideal, approximated)
            for ideal, approximated in zip(
                ideal_points_density_matrices, density_matrices
            )
        ]
    )

    # We also want to plot the displacement vectors. To do so, we compute
    # the displacement for each point.
    U = lat_obtained - lat_ideal
    V = lon_obtained - lon_ideal
    # Make sure all the longitudes are within [-180, 180]
    V[V < -180] += 360
    V[V > 180] -= 360

    lon_ideal, lat_ideal, (Z, U, V) = account_for_periodicity(
        360, lon_ideal, lat_ideal, Z, U, V
    )

    for u, v in zip(U, V):
        complex_number = u + 1.0j * v
        rho = numpy.abs(complex_number)
        angle = numpy.angle(complex_number)
        ax.arrow(0, 0, angle, rho, width=0.002, alpha=0.7)
    # Plot the displacement vectors.
    # ax.quiver(
    #     numpy.zeros_like(Y),
    #     numpy.zeros_like(X),
    #     V,
    #     U,
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minshaft=2,
    #     minlength=0.5,
    #     width=0.002,
    # )

    return fig, ax, cax
