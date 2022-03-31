import typing as ty

import numpy

from sqt import _constants


def bloch_vector_to_density_matrix(s: numpy.ndarray) -> numpy.ndarray:
    """Transform a 3-dimensional point to the corresponding density matrix.

    This method is only valid if the 3-dimensional point is:
    - given in cartesian coordinates.
    - within the Bloch sphere (i.e. 2-norm lower or equal to 1).

    :param s: a 3-dimensional vector representing a point within the unit
        3-dimensional sphere.
    :return: the density matrix corresponding to the state given in parameter
        as a 3-dimensional point.
    """
    return (
        _constants.I + s[0] * _constants.X + s[1] * _constants.Y + s[2] * _constants.Z
    ) / 2


def density2bloch(density_matrix: numpy.ndarray) -> numpy.ndarray:
    """Transform a density matrix to the corresponding Bloch vector.

    This method transforms a **valid** 1-qubit density matrix into a
    3-dimensional cartesian vector within the unit sphere, also known
    as "Bloch vector".

    :param density_matrix: a valid 1-qubit density matrix.
    :return: a 3-dimensional vector representing a point within the unit sphere.
    """
    [[a, b], [c, d]] = density_matrix
    x = complex(c + b).real
    y = complex(c - b).imag
    z = complex(a - d).real
    return numpy.array([x, y, z])


def cartesian2spherical(x: float, y: float, z: float) -> numpy.ndarray:
    """Transform 3-dimensional cartesian coordinates into polar coordinates.

    This function returns the coordinates according to the ISO 80000-2:2019
    convention, meaning that theta is the inclination angle and phi
    is the azimuth angle.

    :param x: the first cartesian coordinate.
    :param y: the second cartesian coordinate.
    :param z: the third cartesian coordinate.
    :return: a 3-dimensional vector [rho, theta, phi] with rho being the
        distance between the given point and the origin (1-norm of the given
        point), theta is the inclination angle and phi is the azimuth angle.
    """
    r = numpy.linalg.norm([x, y, z])
    theta = numpy.arccos(z / r)
    phi = numpy.arctan2(y, x)
    # Physical notation: radius r, inclination theta, azimuth phi
    return numpy.array([r, theta, phi])


def density2spherical(density_matrix: numpy.ndarray) -> numpy.ndarray:
    """Transform a 1-qubit density matrix into spherical coordinates.

    This function returns the spherical coordinates of the Bloch vector
    represented by the given density matrix.

    :param density_matrix: a valid 1-qubit density matrix.
    :return: a 3-dimensional vector [rho, theta, phi] with rho being the
        distance between the given point and the origin (1-norm of the given
        point), theta is the inclination angle and phi is the azimuth angle.
    """
    x, y, z = density2bloch(density_matrix)
    return cartesian2spherical(x, y, z)


def spherical2cartesian(r: float, theta: float, phi: float) -> numpy.ndarray:
    """Transform 3-dimensional polar coordinates into cartesian coordinates.

    :param r: distance between the given point and the origin (1-norm of the
        given point).
    :param theta: the inclination angle.
    :param phi: the azimuth angle.
    :return: a 3-dimensional vector [x, y, z] using cartesian coordinates.
    """
    return numpy.array(
        [
            r * numpy.cos(phi) * numpy.sin(theta),
            r * numpy.sin(phi) * numpy.sin(theta),
            r * numpy.cos(theta),
        ]
    )


def cartesian2density(x: float, y: float, z: float) -> numpy.ndarray:
    """Transform a 3-dimensional point to the corresponding density matrix.

    This method is only valid if the 3-dimensional point is:
    - given in cartesian coordinates.
    - within the Bloch sphere (i.e. 2-norm lower or equal to 1).

    :param s: a 3-dimensional vector representing a point within the unit
        3-dimensional sphere.
    :return: the density matrix corresponding to the state given in parameter
        as a 3-dimensional point.
    """
    coordinates = numpy.array([x, y, z])
    assert (
        abs(numpy.linalg.norm(coordinates)) < 1 + 1e-10
    ), f"The given vector should be in the unit sphere. {coordinates} is not."

    return bloch_vector_to_density_matrix(coordinates)


def spherical2geographic(
    theta: numpy.ndarray, phi: numpy.ndarray
) -> ty.Tuple[numpy.ndarray, numpy.ndarray]:
    lat = 90 - (theta * 360 / (2 * numpy.pi))
    lon = phi * 360 / (2 * numpy.pi)
    lon[lon > 180] -= 360
    return lat, lon
