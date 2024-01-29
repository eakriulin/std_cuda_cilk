# Standard Deviation — CUDA & Cilk

Parallel Computation of Standard Deviation with CUDA and Cilk algorithm implementations.

## Authors

- [Alakbar Damirov](https://github.com/Alis192) — CUDA implementation
- [Eugene Kriulin](https://github.com/eakriulin) — Cilk implementation

## Setup

Prerequisites:

- [Install OpenCilk](https://www.opencilk.org/doc/users-guide/install/)
- // todo: CUDA

Download and enter the project:

```zsh
git clone https://github.com/eakriulin/std_cuda_cilk.git
cd std_cuda_cilk
```

## Run

At each run, the dataset is being generated pseudo randomly. To ensure repeatability, the random seed does not get changed between the runs, which allows to produce the same pseudo random values. It is possible to vary the input size to test the performance.

### CUDA

// todo: CUDA

### Cilk

Run the Cilk version with the [Cilkscale tool](https://www.opencilk.org/doc/users-guide/cilkscale/#how-to-run). You can change the size of the input by varying the value provided in `--args`. For example, the following command tells the program to generate the dataset of 5.000.000 pseudo random values and run the standard deviation algorithm on it:

```zsh
python3 /opt/opencilk/share/Cilkscale_vis/cilkscale.py -c ./stdcilk_cs -b ./stdcilk_cs_benchmark -ocsv ./cstable_std.csv -oplot ./csplot_std.pdf --args 5000000
```
