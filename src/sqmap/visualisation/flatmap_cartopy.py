import warnings
import typing as ty

import numpy
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sqmap.visualisation.flatmap import _compute_infidelity, account_for_periodicity
from sqmap.visualisation.transformers import (
    cartesian2density,
    cartesian2spherical,
)

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes

    def plot_over_projected_bloch_sphere_2d(
        ideal_points_cartesian: ty.List[numpy.ndarray],
        density_matrices: ty.List[numpy.ndarray],
        compute_z_data: ty.Callable[
            [numpy.ndarray, numpy.ndarray], float
        ] = _compute_infidelity,
        projection: ty.Optional[ccrs.Projection] = None,
        fig: ty.Optional[Figure] = None,
        ax: ty.Optional[GeoAxes] = None,
        title: ty.Optional[str] = None,
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
        """
        if projection is None:
            # projection = ccrs.EqualEarth()
            projection = ccrs.Mollweide()
            # projection = ccrs.EqualEarth()
        if fig is None or ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": projection})
            ax.set_global()
        elif not isinstance(ax, GeoAxes):
            raise NotImplementedError(
                "The given ax parameter should be an instance of GeoAxes."
            )

        # Compute the displacement vectors
        ideal_points_spherical = numpy.array(
            [cartesian2spherical(*p) for p in ideal_points_cartesian]
        )
        ideal_points_density_matrices = numpy.array(
            [cartesian2density(*p) for p in ideal_points_cartesian]
        )
        # X and Y are the spherical angles
        X = ideal_points_spherical[:, 1]  # theta, inclination angle
        X = numpy.pi / 2 - X
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

        X, Y, Z = account_for_periodicity(X, Y, Z)
        X *= 180 / numpy.pi
        Y *= 180 / numpy.pi
        # We might not have a lot of points. In order to have a visually
        # good-looking graph, we interpolate the results on a finer "grid"
        # composed of 400 points.
        # The interpolation is linear because we have no idea how this data
        # should behave, so we need to pick the default choice that will
        # likely not mislead us when interpreting the data.
        n = 400
        phi = numpy.linspace(-180, 180, n)
        theta = numpy.linspace(-90, 90, n)
        X_plot, Y_plot = numpy.meshgrid(theta, phi)
        Z_plot = scipy.interpolate.griddata(
            (X, Y), Z, (X_plot, Y_plot), method="linear"
        )

        # Plot each of the ideal points as block dots.
        # Note that X and Y are "swapped" to have an earth-map-like representation
        # of the Bloch sphere.
        ax.plot(Y, X, "k.", ms=1, transform=ccrs.PlateCarree())
        # # Plot the interpolated radial distances on the map as a heatmap.
        filled_c = ax.contourf(
            Y_plot, X_plot, Z_plot, levels=50, transform=ccrs.PlateCarree()
        )

        fig.colorbar(filled_c, orientation="horizontal", format="%1.1e")
        ax.gridlines()
        if title is not None:
            ax.set_title(title)
        return fig, ax


except ImportError:
    warnings.warn(
        "Could not import cartopy. Some functionalities will not be available."
    )
