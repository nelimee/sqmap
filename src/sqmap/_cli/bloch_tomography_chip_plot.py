import argparse
import typing as ty
from pathlib import Path
import pickle

import numpy
import matplotlib.pyplot as plt

from qiskit import IBMQ


from sqmap.visualisation.chip import plot_whole_chip_view


def get_data(
    post_processed_data_backup: Path,
) -> ty.Tuple[
    ty.List[ty.List[ty.Tuple[numpy.ndarray, numpy.ndarray]]], str, str, str, str
]:
    with open(post_processed_data_backup, "rb") as f:
        data = pickle.load(f)

    return (
        data["density_matrices"],
        data["backend_name"],
        data["basis_name"],
        data["post_processing_method"],
        data["provider"],
    )


def get_processor_type(backend) -> str:
    processor_type_data = backend.configuration().processor_type
    processor_type: str = (
        processor_type_data.get("family", "")
        + " r"
        + str(processor_type_data.get("revision", ""))
        + processor_type_data.get("segment", "")
    )
    return processor_type


def main():
    parser = argparse.ArgumentParser(
        description=("Plot the reconstructed density matrices.")
    )
    parser.add_argument(
        "backup_file",
        type=Path,
        help="Backup file path that has been saved after computing the density matrices.",
    )
    args = parser.parse_args()

    (
        density_matrices,
        backend_name,
        basis_name,
        post_processing_method,
        provider_data,
    ) = get_data(args.backup_file)

    if not IBMQ.active_account():
        print("Loading IBMQ account, this might take some time...")
        IBMQ.load_account()
    print("Recovering provider and backend data...")
    provider = IBMQ.get_provider(**provider_data)
    backend = provider.get_backend(backend_name)

    processor_type: str = get_processor_type(backend)

    plot_whole_chip_view(density_matrices, processor_type)
    plt.show()
