# Lattice Resonances.

______________________________________________________________________

To use this package install the [Poetry](https://python-poetry.org/) package manager, and run `poetry install`.

The `main.py` currently accepts two command-line arguments, a path to a specification file (examples are in the `inputs` subdirectory) and a periodicity in nanometres. For example, running the following command:

```
    python main.py inputs/tm_sic.toml 15000
```

will run the calculation for the file `tm_sic.toml` with periodicity 15 microns. The main function calculates and plots the cross sections.

Some useful example scripts are available in the `/scripts`. These calculate the response for infinite arrays, a useful comparator.

## Suggestions

______________________________________________________________________

- The code in this package is based on two Papers, 10.1109/JPROC.2019.2939396 and 10.1021/acs.accounts.9b00312. As a good first step, it would be smart to verify the observables match these results. I have started this process for the former paper, scripts to generate the figures from this work are available in the `verify` subdirectory, but these should be checked thoroughly for correctness.

- To understand how the coupling evolves from small to infinite arrays the array size could be scanned from small (5 x 5) to a value large enough to achieve convergence between the finite and infinite spectrums.

- To understand how pillar spacing affects the coupling the array period could be tuned from a small value (5um say) to a large one. At some point the strong coupling will disappear, as SPPs are unable to propagate significant energy between pillars in the array.

* Different pillars could be used in the sums to evaluate observables. By default we sum over the whole array. I have introduced an `Enum` describing the probe region (in `types/field.py`) with other options (`Corner`, `Centre`, `Edge`). At the moment these take the signal from one pillar, which does not seem to give interesting data. By changing this code to instead sum over a small number of pillars in the region of interest, the change in response across the array could be calculated.
