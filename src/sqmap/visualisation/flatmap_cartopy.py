import typing as ty
import warnings

import matplotlib.pyplot as plt
import numpy
import scipy.interpolate
from matplotlib.figure import Figure
from sqmap.visualisation.flatmap import _compute_infidelity, account_for_periodicity
from sqmap.visualisation.transformers import (
    cartesian2density,
    cartesian2spherical,
    density2spherical,
    spherical2geographic,
)

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes

    def plot_over_projected_bloch_sphere_2d(
        ideal_points_cartesian: list[numpy.ndarray],
        density_matrices: list[numpy.ndarray],
        compute_z_data: ty.Callable[
            [numpy.ndarray, numpy.ndarray], float
        ] = _compute_infidelity,
        projection: ccrs.Projection | None = None,
        fig: Figure | None = None,
        ax: GeoAxes | None = None,
        title: str | None = None,
        figure_subplots_indices: tuple[int, int] | None = None,
        ax_index: int | None = None,
    ):
        """Plot the data computed by compute_z_data on a 2-dimensional heatmap.

        TODO: check that all the angles are translated correctly!!

        :param ideal_points_cartesian: the points that have been prepared using quantum
            gates and that should ideally be retrieved via state tomography.
            Should be in cartesian coordinates.
        :param density_matrices: a list of density matrices that should have as many
            elements as the ideal_points_cartesian parameter.
        :param compute_z_data: a function that takes 2 density matrices as parameters
            (the ideal density matrix as first parameter and the measured one as second
            parameter) and returns a float value that will be plotted on the flattened
            Bloch sphere.
        :param projection: the projection used for the map.
        :param fig: the figure to use. If either fig or ax is None, fig **and** ax are
            obtained through a call to plt.subplots().
        :param ax: the ax to use. If either fig or ax is None, fig **and** ax are
            obtained through a call to plt.subplots().
        :param title: title of the subplot.
        :param figure_subplots_indices: if ax and fig are provided, should be the
            tuple (x, y) if the fig and ax have been created like that:
                fig, axes = plt.subplots(x, y)
        :param ax_index: if ax and fig are provided, should be the position of
            the ax on the given figure. This is **not** the position returned by
            ax.get_position() but rather the index of the ax when created with
            plt.subplots.
        """
        if projection is None:
            projection = ccrs.PlateCarree()
            # projection = ccrs.EqualEarth()
            # projection = ccrs.Mollweide()

        if fig is None or ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": projection})
            ax.set_global()
        elif (
            not isinstance(ax, GeoAxes)
            and figure_subplots_indices is not None
            and ax_index is not None
        ):
            fig.delaxes(ax)
            ax = fig.add_subplot(
                *figure_subplots_indices, ax_index, projection=projection
            )
        else:
            raise NotImplementedError(
                "Case not implemented. Read the docstring and make sure "
                "that you check the conditions described."
            )
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
            lon_ideal, lat_ideal, Z, U, V
        )

        # Plot each of the ideal points as black dots.
        ax.plot(lon_ideal, lat_ideal, "k.", ms=1, transform=ccrs.PlateCarree())

        # We might not have a lot of points. In order to have a visually
        # good-looking graph, we interpolate the results on a finer "grid".
        # The interpolation is linear because we have no idea how this data
        # should behave, so we need to pick the default choice that will
        # likely not mislead us when interpreting the data.
        n = 1000
        phi_interp = numpy.linspace(-180, 180, n)
        theta_interp = numpy.linspace(-90, 90, n)
        X_plot, Y_plot = numpy.meshgrid(theta_interp, phi_interp)
        Z_plot = scipy.interpolate.griddata(
            (lat_ideal, lon_ideal), Z, (X_plot, Y_plot), method="linear"
        )

        # Plot the interpolated radial distances on the map as a heatmap.
        filled_c = ax.contourf(
            Y_plot, X_plot, Z_plot, levels=50, transform=ccrs.PlateCarree()
        )
        ax.quiver(
            lon_ideal,
            lat_ideal,
            V,
            U,
            transform=ccrs.PlateCarree(),
            angles="xy",
            scale_units="xy",
            scale=1,
            minshaft=2,
            minlength=0.5,
            width=0.002,
        )
        fig.colorbar(filled_c, orientation="horizontal", format="%1.1e")
        ax.gridlines(draw_labels="x")
        if title is not None:
            ax.set_title(title)

        return fig, ax


except ImportError:
    warnings.warn(
        "Could not import cartopy. Some functionalities will not be available."
    )
