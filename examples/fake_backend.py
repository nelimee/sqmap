import typing as ty

import numpy
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.mle import post_process_tomography_results_mle

from sqt.basis.equidistant import get_approximately_equidistant_circuits
from sqmap.visualisation.flatmap import plot_bloch_vector_displacement_arrow_field_2d

# %%
hub = "ibm-q-lanl"
group = "lanl"
project = "quantum-simulati"
backend_name = "ibm_algiers"

print(f"Recovering IBMQ backend {backend_name} with {hub}/{group}/{project}.")
service = QiskitRuntimeService(
    channel="ibm_quantum", instance=f"{hub}/{group}/{project}"
)
if not service.active_account():
    raise RuntimeError(f"Could not load account with '{hub}' '{group}' '{project}'.")
backend = service.get_backend("ibm_algiers")

# %%
print(f"Initialising a noisy simulator with {backend_name} calibrations.")
simulator = AerSimulator.from_backend(backend)
qubit_number: int = 1
max_shots: int = 2**15
nb_points_on_bloch_sphere: int = 100

circuits = get_approximately_equidistant_circuits(nb_points_on_bloch_sphere)

print("Generating tomography circuits...")
basis = TetrahedralMeasurementBasis()
tomography_circuits = [
    one_qubit_tomography_circuits(circuit, basis, qubit_number) for circuit in circuits
]

# %%
print("Simulating tomography circuits...")
result = simulator.run(sum(tomography_circuits, start=[]), shots=2**13).result()

# %%
print("Computing the density matrices...")
density_matrices_all_qubits: ty.List[ty.List[numpy.ndarray]] = [
    post_process_tomography_results_mle(result, circuit, basis, qubit_number)
    for circuit in circuits
]

# %%
print("Plotting...")
qubit_index = 0
points: ty.List[numpy.ndarray] = list()
density_matrices: ty.List[numpy.ndarray] = list()
for i, circuit in enumerate(circuits):
    point: numpy.ndarray = numpy.asarray(eval(circuit.name))  # UGLY
    points.append(point)
    density_matrices.append(density_matrices_all_qubits[i][qubit_index])

plot_bloch_vector_displacement_arrow_field_2d(points, density_matrices)
plt.show()

# %%
