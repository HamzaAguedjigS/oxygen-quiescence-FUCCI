# Oxygen-regulated cell-cycle model with hidden quiescence

This repository contains the computational material supporting the article
"A hidden quiescent reservoir shapes extinction thresholds and limits
FUCCI-based inference in an oxygen-regulated cell-cycle model."

The model tracks cycling G1 cells (`R`), S/G2/M cells (`G`), reversibly
quiescent cells (`Q`) and oxygen (`N`). Time is measured in hours throughout
the code, and the state order is always `(R, G, Q, N)`.

## Repository structure

- `code/`: model definition, analysis scripts and figure-generation scripts.
- `results/`: machine-readable outputs produced with fixed random seeds.
- `figures/`: final vector PDFs and high-resolution PNG renderings generated
  from the archived outputs; `fig1_model.pdf` is the model schematic.

The repository contains no patient-level, animal or personally identifiable
data. All numerical experiments are synthetic and use the parameter regime
specified in the article.

## Software environment

Python 3.9 or later is recommended. For a lightweight installation, create a
virtual environment and install the required packages with:

```text
python -m pip install -r requirements.txt
```

To reproduce the exact tested environment with Conda, run:

```text
conda env create -f environment.yml
conda activate oxygen-quiescence-fucci
```

The archived JSON and NPZ files allow the figures and manuscript macros to be
regenerated without rerunning the longest Monte Carlo and stability scans.
The reproduction commands below were tested with Python 3.9, NumPy 1.21.5,
SciPy 1.9.1, SymPy 1.10.1 and Matplotlib 3.5.2.

## Reproduce manuscript figures and numerical macros

From the repository root, run:

```text
python code/make_macros.py
python code/make_figures.py
```

These commands read the archived files in `results/` and update the generated
outputs in `results/` and `figures/`. They do not rerun the computationally
intensive analyses.

## Recompute the analyses

The principal analysis scripts are:

- `p0_parameter_candidates.py`: reference-regime search.
- `p1_stability_scan.py`: local-stability parameter scan.
- `p1_global_recheck.py`: global trajectory check with fixed initial states.
- `p1_hopf_example.py` and `p1_hopf_analysis.py`: numerical Hopf example.
- `p1_sotomayor.py`: numerical verification of transcritical coefficients.
- `p1_identifiability.py`: exact Taylor-rank calculation.
- `p2_dynamics.py`: threshold, reservoir and control calculations.
- `p2_observability.py`: reoxygenation observation calculations.
- `p2_montecarlo.py reference`: reference-vector Monte Carlo experiments.
- `p2_montecarlo.py ensemble`: parameter-ensemble Monte Carlo experiments.
- `p2_morris.py`: global sensitivity screen.
- `p5_exposure_boundary.py`: near-boundary exposure calculations.

Several calculations are intentionally computationally intensive. Each script
writes its outputs to `results/`; random seeds and solver tolerances are fixed
in the scripts and recorded in the corresponding machine-readable outputs.
The archived outputs are therefore sufficient for reproducing every numerical
figure without repeating the longest simulations.

## Citation and archival release

Please cite the associated article when using this material. Citation metadata
are provided in `CITATION.cff`, which GitHub displays through its
"Cite this repository" interface. Add the repository DOI to this section and
to `CITATION.cff` after archiving the first public release with Zenodo.

## Contact

Hamza Aguedjig — hamza.aguedjig@um5r.ac.ma
