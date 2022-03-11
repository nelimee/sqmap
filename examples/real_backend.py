import os
from pathlib import Path
import pickle
import typing as ty

import numpy
import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit.circuit.quantumcircuit import QuantumCircuit

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.basis.equidistant import get_approximately_equidistant_circuits
from sqt.passes import compile_circuits

from sqmap.visualisation.flatmap import plot_bloch_vector_displacement_arrow_field_2d


try:
    HERE = Path(__file__).parent
except:
    HERE = Path("/home/suau/lanl/sqmap/examples/")
BACKUP = HERE / "backups"
if not BACKUP.is_dir():
    os.makedirs(BACKUP)

n: int = 70

circuits = get_approximately_equidistant_circuits(n)
for c in circuits:
    c.barrier()

print("Recovering data from IBMQ...")
if not IBMQ.active_account():
    IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-lanl", group="lanl", project="quantum-simulati")
backend = provider.get_backend("ibmq_guadalupe")

max_shots: int = backend.configuration().max_shots

print("Generating tomography circuits...")
qubit_number: int = backend.configuration().num_qubits
basis = TetrahedralMeasurementBasis()
tomography_circuits = [
    one_qubit_tomography_circuits(circuit, basis, qubit_number) for circuit in circuits
]

submitted_circuits: ty.List[QuantumCircuit] = compile_circuits(
    sum(tomography_circuits, start=[])
)

print("Simulating tomography circuits...")
# job_id: str = "6222802d317810cb4c202352"  # ibm_auckland,   rep_delay=0.00025
job_id: str = "6225ce2ecd47037d7be99c6c"  # ibmq_guadalupe, rep_delay=0.00025
# job_id: str = "6225d39b827b3771fa803375"  # ibm_auckland,   rep_delay=0.00001
# job_id: str = "6225d45e3178105c8f20331a"  # ibm_washington, rep_delay=0.00001
# job_id: str = "6225d925eee8da4c32567905"  # ibmq_guadalupe, rep_delay=0.00001
# job_id: str = None
if job_id is not None:
    print(f"Retrieving job '{job_id}'")

    with open(BACKUP / f"{job_id}.pkl", "rb") as f:
        result = pickle.load(f)
else:
    job = backend.run(
        submitted_circuits,
        shots=max_shots,
        job_name=f"bloch_sphere_{n}",
        job_tags=["tomography", "bloch_sphere", f"points={n}", f"basis={basis.name}"],
        init_qubits=True,
        # rep_delay=0.00025,  # default
        rep_delay=0.00001,
    )
    result = job.result()
    with open(BACKUP / f"{job.job_id()}.pkl", "wb") as f:
        pickle.dump(result, f)

print("Computing the density matrices...")
density_matrices_all_qubits: ty.List[ty.List[numpy.ndarray]] = [
    post_process_tomography_results_mle(result, circuit, basis, qubit_number)
    for circuit in circuits
]

print("Plotting...")
for qubit_index in range(1):
    points: ty.List[numpy.ndarray] = list()
    density_matrices: ty.List[numpy.ndarray] = list()
    for i, circuit in enumerate(circuits):
        point: numpy.ndarray = numpy.asarray(eval(circuit.name))  # UGLY
        points.append(point)
        density_matrices.append(density_matrices_all_qubits[i][qubit_index])
    plot_bloch_vector_displacement_arrow_field_2d(points, density_matrices)
plt.show()
