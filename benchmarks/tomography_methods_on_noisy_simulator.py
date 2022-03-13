import typing as ty
import numpy
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.states import state_fidelity

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.fit.lssr import post_process_tomography_results_lssr
from sqt.fit.exact import get_one_qubit_exact_density_matrix
from sqt.execution import execute
from sqt.basis.equidistant import get_approximately_equidistant_circuits

from sqmap.visualisation.flatmap import plot_over_bloch_sphere_2d

print("Recovering data from IBMQ...")
if not IBMQ.active_account():
    IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-lanl", group="lanl", project="quantum-simulati")
backend = provider.get_backend("ibmq_bogota")


N = 10
METHODS = {
    "grad": post_process_tomography_results_grad,
    "mle": post_process_tomography_results_mle,
    "lssr": post_process_tomography_results_lssr,
}
simulator = AerSimulator.from_backend(backend)

print("Creating circuits...")
raw_circuits: ty.List[QuantumCircuit] = get_approximately_equidistant_circuits(N)

print("Creating tomography circuits...")
qubit_number = backend.configuration().num_qubits
basis = TetrahedralMeasurementBasis()
tomography_circuits: ty.List[ty.List[QuantumCircuit]] = [
    one_qubit_tomography_circuits(raw_circuit, basis, qubit_number=qubit_number)
    for raw_circuit in raw_circuits
]
flattened_circuits: ty.List[QuantumCircuit] = sum(tomography_circuits, start=[])

print("Simulating circuits...")
result = execute(flattened_circuits, simulator, shots=2 ** 10)

print("Computing density matrices...")
density_matrices: ty.List[ty.Dict[str, ty.List[numpy.ndarray]]] = list()
fidelities: ty.List[ty.Dict[str, ty.List[float]]] = list()
points: ty.List[numpy.ndarray] = list()
for circuit in raw_circuits:
    points.append(numpy.array(eval(circuit.name)))  # HHHHAAAAAAAA
    exact_density_matrix: numpy.ndarray = get_one_qubit_exact_density_matrix(circuit)
    density_matrices.append(
        {"exact": [exact_density_matrix.copy() for _ in range(qubit_number)]}
    )
    fidelities.append({"exact": [1.0 for _ in range(qubit_number)]})
    for method in METHODS:
        density_matrices[-1][method] = METHODS[method](
            result, circuit, basis, qubit_number=qubit_number
        )

        fidelities[-1][method] = [
            state_fidelity(exact, approximation, validate=False)
            for exact, approximation in zip(
                density_matrices[-1]["exact"], density_matrices[-1][method]
            )
        ]

print("Plotting...")
for qubit_index in range(qubit_number):
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for method, ax in zip(METHODS, axes):
        plot_over_bloch_sphere_2d(
            points,
            [dm[method][qubit_index] for dm in density_matrices],
            fig=fig,
            ax=ax,
            title=f"{method}",
        )
plt.show()
