"""Module to help visualising tomography results over a whole backend."""

from dataclasses import dataclass
import typing as ty
import warnings

import matplotlib.pyplot as plt


@dataclass
class QubitPlacement:
    qubit_number: int
    positions: ty.List[ty.Tuple[int, int]]
    supported_processor_types: ty.List[str]

    def __post_init__(self):
        if not self.qubit_number == len(self.positions):
            raise RuntimeError(
                f"Error when initialising {type(self).__name__}: {self.qubit_number} "
                f"qubits are needed but only {len(self.positions)} positions are given."
            )

    def plot(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(
            [p[0] for p in self.positions], [p[1] for p in self.positions], "k.", ms=3
        )
        ax.invert_yaxis()

    @property
    def max_x(self) -> int:
        return max(p[0] for p in self.positions)

    @property
    def max_y(self) -> int:
        return max(p[1] for p in self.positions)


class OneQubit(QubitPlacement):
    def __init__(self):
        super().__init__(
            qubit_number=1,
            positions=[(0, 0)],
            supported_processor_types=["Canary r1.2"],
        )


class TShape5Qubits(QubitPlacement):
    def __init__(self):
        super().__init__(
            qubit_number=5,
            positions=[(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
            supported_processor_types=["Falcon r4T"],
        )


class LineShape5Qubits(QubitPlacement):
    def __init__(self):
        super().__init__(
            qubit_number=5,
            positions=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            supported_processor_types=["Falcon r4L", "Falcon r5.11L"],
        )


class HShape7Qubits(QubitPlacement):
    def __init__(self):
        super().__init__(
            qubit_number=7,
            positions=[(0, 0), (1, 0), (2, 0), (1, 1), (0, 2), (1, 2), (2, 2)],
            supported_processor_types=["Falcon r5.11H"],
        )


class Shape16Qubits(QubitPlacement):
    def __init__(self):
        super().__init__(
            qubit_number=16,
            positions=[
                (0, 1),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 3),
                (3, 4),
                (4, 1),
                (4, 3),
                (5, 1),
                (5, 2),
                (5, 3),
                (6, 1),
            ],
            supported_processor_types=["Falcon r4P"],
        )


class Shape27Qubits(QubitPlacement):
    def __init__(self):
        super().__init__(
            qubit_number=27,
            positions=[
                (0, 1),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 3),
                (3, 4),
                (4, 1),
                (4, 3),
                (5, 1),
                (5, 2),
                (5, 3),
                (6, 1),
                (6, 3),
                (7, 0),
                (7, 1),
                (7, 3),
                (7, 4),
                (8, 1),
                (8, 3),
                (9, 1),
                (9, 2),
                (9, 3),
                (10, 3),
            ],
            supported_processor_types=[
                "Falcon r5.11",
                "Falcon r4",
                "Falcon r5.1",
                "Falcon r8",
            ],
        )


class Shape65Qubits(QubitPlacement):
    def __init__(self):
        positions: ty.List[ty.Tuple[int, int]] = [(i, 0) for i in range(10)]
        positions += [(0, 1), (4, 1), (8, 1)]
        positions += [(i, 2) for i in range(11)]
        positions += [(2, 3), (6, 3), (10, 3)]
        positions += [(i, 4) for i in range(11)]
        positions += [(0, 5), (4, 5), (8, 5)]
        positions += [(i, 6) for i in range(11)]
        positions += [(2, 7), (6, 7), (10, 7)]
        positions += [(i, 8) for i in range(1, 11)]

        super().__init__(
            qubit_number=65,
            positions=positions,
            supported_processor_types=["Hummingbird r2"],
        )


class Shape127Qubits(QubitPlacement):
    def __init__(self):
        positions: ty.List[ty.Tuple[int, int]] = [(i, 0) for i in range(14)]
        positions += [(0, 1), (4, 1), (8, 1), (12, 1)]
        positions += [(i, 2) for i in range(15)]
        positions += [(2, 3), (6, 3), (10, 3), (14, 3)]
        positions += [(i, 4) for i in range(15)]

        positions += [(0, 5), (4, 5), (8, 5), (12, 5)]
        positions += [(i, 6) for i in range(15)]
        positions += [(2, 7), (6, 7), (10, 7), (14, 7)]
        positions += [(i, 8) for i in range(15)]

        positions += [(0, 9), (4, 9), (8, 9), (12, 9)]
        positions += [(i, 10) for i in range(15)]
        positions += [(2, 11), (6, 11), (10, 11), (14, 11)]
        positions += [(i, 12) for i in range(1, 15)]

        super().__init__(
            qubit_number=127,
            positions=positions,
            supported_processor_types=["Eagle r1"],
        )


_PLACEMENTS: ty.Dict[str, QubitPlacement] = {}
for placement_class in [
    OneQubit,
    TShape5Qubits,
    LineShape5Qubits,
    HShape7Qubits,
    Shape16Qubits,
    Shape27Qubits,
    Shape65Qubits,
    Shape127Qubits,
]:
    instance: QubitPlacement = placement_class()
    for processor_type in instance.supported_processor_types:
        if processor_type in _PLACEMENTS:
            warnings.warn(
                f"Processor type '{processor_type}' is handled by more than one QubitPlacement "
                + "subclass: "
                + placement_class.__name__
                + ", "
                + type(_PLACEMENTS[processor_type]).__name__
                + "."
            )
        _PLACEMENTS[processor_type] = placement_class()


def get_qubit_placement(processor_type: str) -> QubitPlacement:
    return _PLACEMENTS[processor_type]
