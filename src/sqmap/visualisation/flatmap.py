import typing as ty

import numpy
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qiskit.quantum_info.states import DensityMatrix, state_fidelity

from sqmap.visualisation.transformers import (
    cartesian2density,
    density2spherical,
    cartesian2spherical,
)


def _compute_infidelity(ideal: numpy.ndarray, approximated: numpy.ndarray) -> float:
    return 1 - state_fidelity(
        DensityMatrix(ideal), DensityMatrix(approximated), validate=False
    )


def account_for_periodicity(
    X: numpy.ndarray, Y: numpy.ndarray, Z: numpy.ndarray
) -> ty.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    # For angles that are very close to -pi or pi, we duplicate the values in order to
    # have a "kind of periodicity" when plotting.
    (indices_minus_pi,) = numpy.nonzero(numpy.isclose(Y, -numpy.pi))
    (indices_plus_pi,) = numpy.nonzero(numpy.isclose(Y, numpy.pi))
    x = numpy.concatenate([X, X[indices_minus_pi], X[indices_plus_pi]])
    y = numpy.concatenate(
        [Y, Y[indices_minus_pi] + 2 * numpy.pi, Y[indices_plus_pi] - 2 * numpy.pi]
    )
    z = numpy.concatenate([Z, Z[indices_minus_pi], Z[indices_plus_pi]])
    return x, y, z


def plot_over_bloch_sphere_2d(
    ideal_points_cartesian: ty.List[numpy.ndarray],
    density_matrices: ty.List[numpy.ndarray],
    compute_z_data: ty.Callable[
        [numpy.ndarray, numpy.ndarray], float
    ] = _compute_infidelity,
    title: ty.Optional[str] = None,
    fig=None,
    ax=None,
    cax=None,
    clabel: ty.Optional[str] = None,
    vmin: float = None,
    vmax: float = None,
):
    """Plot the data computed by compute_z_data on a 2-dimensional heatmap.

    :param ideal_points_cartesian: the points that have been prepared using quantum
        gates and that should ideally be retrieved via state tomography.
        Should be in cartesian coordinates.
    :param density_matrices: a list of density matrices that should have as many
        elements as the ideal_points_cartesian parameter.
    :param compute_z_data: a function that takes 2 density matrices as parameters
        (the ideal density matrix as first parameter and the measured one as second
        parameter) and returns a float value that will be plotted on the flattened
        Bloch sphere.
    :param fig: the figure to use. If either fig or ax is None, fig **and** ax are
        obtained through a call to plt.subplots().
    :param ax: the ax to use. If either fig or ax is None, fig **and** ax are
        obtained through a call to plt.subplots().
    :param cax: the ax used to plot the colorbar. If None, the ax is constructed.
    :param vmin: minimum value for the colorbar.
    :param vmax: maximum value for the colorbar.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    # Compute the displacement vectors
    ideal_points_spherical = numpy.array(
        [cartesian2spherical(*p) for p in ideal_points_cartesian]
    )
    ideal_points_density_matrices = numpy.array(
        [cartesian2density(*p) for p in ideal_points_cartesian]
    )
    # X and Y are the spherical angles
    X = ideal_points_spherical[:, 1]  # theta, inclination angle
    Y = ideal_points_spherical[:, 2]  # phi,   azimuth angle
    # The colormap will plot the infidelity of the reconstructed state, i.e.
    # 1 minus the fidelity between the ideal state and the reconstructed one.
    Z = numpy.array(
        [
            compute_z_data(ideal, approximated)
            for ideal, approximated in zip(
                ideal_points_density_matrices, density_matrices
            )
        ]
    )

    average_value: float = numpy.mean(Z)
    X, Y, Z = account_for_periodicity(X, Y, Z)
    # We might not have a lot of points. In order to have a visually
    # good-looking graph, we interpolate the results on a finer "grid"
    # composed of 400 points.
    # The interpolation is linear because we have no idea how this data
    # should behave, so we need to pick the default choice that will
    # likely not mislead us when interpreting the data.
    n = 400
    phi = numpy.linspace(-numpy.pi, numpy.pi, n)
    theta = numpy.linspace(0, numpy.pi, n)
    X_plot, Y_plot = numpy.meshgrid(theta, phi)
    Z_plot = scipy.interpolate.griddata((X, Y), Z, (X_plot, Y_plot), method="linear")

    # Plot each of the ideal points as block dots.
    # Note that X and Y are "swapped" to have an earth-map-like representation
    # of the Bloch sphere.
    ax.plot(Y, X, "k.", ms=1)
    # Plot the interpolated radial distances on the map as a heatmap.
    c = ax.imshow(
        Z_plot.T, extent=(-numpy.pi, numpy.pi, 0, numpy.pi), vmin=vmin, vmax=vmax
    )

    ax.invert_yaxis()
    ax.set_xlabel(r"$\phi$")
    ax.set_xlim(left=-numpy.pi, right=numpy.pi)
    ax.set_ylabel(r"$\theta$")
    ax.set_ylim(top=0, bottom=numpy.pi)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    if clabel is not None:
        cax.set_title(clabel)
    colorbar = fig.colorbar(c, cax=cax)
    colorbar.ax.hlines(average_value, 0, 1, color="r")
    ax.set_title(title)
    return fig, ax, cax


def plot_bloch_vector_displacement_arrow_field_2d(
    ideal_points_cartesian: ty.List[numpy.ndarray],
    density_matrices: ty.List[numpy.ndarray],
    compute_z_data: ty.Callable[
        [numpy.ndarray, numpy.ndarray], float
    ] = _compute_infidelity,
    title: ty.Optional[str] = None,
    fig=None,
    ax=None,
    cax=None,
    vmin: float = None,
    vmax: float = None,
):
    """Plot a 2-dimensional heatmap with displacement arrows.

    The displacement arrows that represent the displacement between the ideally
    prepared state and the reconstructed one are plotted above the heatmap that
    represents the 2-norm of the reconstructed bloch vector.

    TODO: check that there is no mistake with theta/phi and that the map that is
          plotted can be seen as an earth-map like, i.e. the north pole
          (state |0>) is at the top of the plot, the south pole (state |1>) is at
          the bottom of the plot, and all the equally superposed states are on
          the equator.
          See plot_bloch_vector_state_infidelity_2d that has been checked and is
          correct with respect to theta and phi.
    :param fig: the figure to use. If either fig or ax is None, fig **and** ax are
        obtained through a call to plt.subplots().
    :param ax: the ax to use. If either fig or ax is None, fig **and** ax are
        obtained through a call to plt.subplots().
    :param cax: the ax used to plot the colorbar. If None, the ax is constructed.
    :param vmin: minimum value for the colorbar.
    :param vmax: maximum value for the colorbar.
    """
    fig, ax, cax = plot_over_bloch_sphere_2d(
        ideal_points_cartesian,
        density_matrices,
        compute_z_data=compute_z_data,
        title=title,
        fig=fig,
        ax=ax,
        cax=cax,
        vmin=vmin,
        vmax=vmax,
    )
    # Compute the displacement vectors
    obtained_points_spherical = numpy.array(
        [density2spherical(rho) for rho in density_matrices]
    )
    ideal_points_spherical = numpy.array(
        [cartesian2spherical(*p) for p in ideal_points_cartesian]
    )

    # X and Y are the spherical angles
    X = ideal_points_spherical[:, 1]  # theta, inclination angle
    Y = ideal_points_spherical[:, 2]  # phi,   azimuth angle

    # We also want to plot the displacement vectors. To do so, we compute
    # the displacement for each point.
    U = obtained_points_spherical[:, 1] - ideal_points_spherical[:, 1]
    V = obtained_points_spherical[:, 2] - ideal_points_spherical[:, 2]
    # Make sure all the points are within [-pi, pi]
    U[U < -numpy.pi] += 2 * numpy.pi
    U[U > numpy.pi] -= 2 * numpy.pi
    V[V < -numpy.pi] += 2 * numpy.pi
    V[V > numpy.pi] -= 2 * numpy.pi

    _, _, U = account_for_periodicity(X, Y, U)
    X, Y, V = account_for_periodicity(X, Y, V)

    # Plot the displacement vectors.
    ax.quiver(
        Y,
        X,
        V,
        U,
        angles="xy",
        scale_units="xy",
        scale=1,
        minshaft=2,
        minlength=0.5,
        width=0.002,
    )

    return fig, ax, cax
