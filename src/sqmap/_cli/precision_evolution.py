import typing as ty
from pathlib import Path
import pickle
import numpy
import argparse
import matplotlib.pyplot as plt

import qiskit.quantum_info as qi

from sqmap.visualisation.transformers import cartesian2density


def get_data(
    datafiles: ty.List[Path],
    metric_method: ty.Callable[[numpy.ndarray, numpy.ndarray], float],
) -> ty.Dict[str, ty.Dict[int, ty.List[float]]]:
    metric_values: ty.Dict[str, ty.Dict[int, ty.List[float]]] = {}
    basis_names: ty.Set[str] = set()
    for path in datafiles:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data["qubit_number"] != 1:
            print("Warning: found a file with", data["qubit_number"], "qubits.")
        qubit_index: int = 0
        basis_names.add(data["basis_name"])
        post_processing_method_name: str = data["post_processing_method"]
        if post_processing_method_name not in metric_values:
            metric_values[post_processing_method_name] = dict()
        shots: int = data["shots"]
        metrics: ty.List[float] = [
            metric_method(cartesian2density(*ideal_point), obtained)
            for ideal_point, obtained in data["density_matrices"][qubit_index]
        ]
        metric_values[post_processing_method_name][shots] = metrics
    if len(basis_names) > 1:
        print("Found several basis:", ", ".join(basis_names))
    return metric_values


def plot(
    datafiles: ty.List[Path],
    metric_method: ty.Callable[[numpy.ndarray, numpy.ndarray], float],
    metric_name: str,
    method: str,
) -> None:
    metric_values = get_data(datafiles, metric_method)
    for post_processing_method_name, metric_data in metric_values.items():
        X, Y = [], []
        for shots, metric_values in sorted(metric_data.items()):
            summary_metric: float = getattr(numpy, method)(metric_values)
            X.append(shots)
            Y.append(summary_metric)
        Y = numpy.real_if_close(Y)
        plt.loglog(X, Y, label=post_processing_method_name)
    plt.xlabel("Number of shots used")
    plt.ylabel(metric_name)
    plt.title(
        f"Plot of the obtained {method} {metric_name} for a varying\n"
        "number of shots performed on an ideal simulator."
    )
    plt.grid()
    plt.legend()


_METRICS = {
    "fidelity": lambda ideal, obtained: qi.state_fidelity(
        ideal, obtained, validate=False
    ),
    "infidelity": lambda ideal, obtained: 1
    - qi.state_fidelity(ideal, obtained, validate=False),
    "purity": lambda ideal, obtained: qi.purity(obtained, validate=False),
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the obtained precision for different number of shots "
            "and different post-processing methods."
        )
    )
    parser.add_argument(
        "datafiles",
        type=Path,
        nargs="+",
        help="Files containing the data that will be plotted.",
    )
    parser.add_argument(
        "--metric",
        help="Use the metric to use.",
        choices=_METRICS.keys(),
        default="fidelity",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="mean",
        help="Name of the numpy function used to summarise the data.",
    )

    args = parser.parse_args()

    metric_method = _METRICS[args.metric]
    plot(args.datafiles, metric_method, args.metric, args.summary)
    plt.show()
