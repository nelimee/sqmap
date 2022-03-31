# Single Qubit MAPping

This project has been made during the Quantum Computing Summer School 2021 at Los Alamos National Laboratory with the help of:

- Carleton Coffrin
- Marc Vuffray
- Andrey Lokhov
- Jon Nelson

and continued during my PhD.

## Quick start

### Install

In order to start using the code in this repository, first install the `sqmap` package with the command
```sh
git clone git@github.com/nelimee/sqmap
python -m pip install -e sqmap/
```

### Command line interface

The `sqmap` package provides scripts that are installed along with the package.

#### `sqmap_bloch_tomography_plot`

This script plots the results of a given tomography experiment.

```
>>> sqmap_bloch_tomography_plot --help
usage: sqmap_bloch_tomography_plot [-h] [-i QUBIT_INDEX] backup_file

Plot the reconstructed density matrices.

positional arguments:
  backup_file           Backup file path that has been saved after computing the density
                        matrices.

optional arguments:
  -h, --help            show this help message and exit
  -i QUBIT_INDEX, --qubit-index QUBIT_INDEX
                        Index of the qubit to plot. Default to all qubits if not given.
```



### Using `sqmap`
