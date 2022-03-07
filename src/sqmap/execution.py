import typing as ty

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet, ManagedResults


def execute(
    circuits: ty.List[QuantumCircuit],
    backend: IBMQBackend,
    job_name: ty.Optional[str] = None,
    tags: ty.Optional[ty.List[str]] = None,
    **kwargs,
) -> Result:
    """Execute the given circuits on the backend.

    :param circuits: quantum circuit instances to execute on the given backend.
    :param backend: backend used to execute the given circuits.
    :param job_name: prefix used for the job name. IBMQJobManager will add
        a suffix for each job.
    :param tags: tags for each of the submitted jobs.
    :param kwargs: forwarded to run_config.
        Configuration of the runtime environment. Some
        examples of these configuration parameters include:
        ``qobj_id``, ``qobj_header``, ``shots``, ``memory``,
        ``seed_simulator``, ``qubit_lo_freq``, ``meas_lo_freq``,
        ``qubit_lo_range``, ``meas_lo_range``, ``schedule_los``,
        ``meas_level``, ``meas_return``, ``meas_map``,
        ``memory_slot_size``, ``rep_time``, and ``parameter_binds``.

        Refer to the documentation on :func:`qiskit.compiler.assemble`
        for details on these arguments.
    """
    manager = IBMQJobManager()
    job: ManagedJobSet = manager.run(
        circuits, backend, name=job_name, job_tags=tags, run_config=kwargs
    )
    results: ManagedResults = job.results()
    return results.combine_results()
