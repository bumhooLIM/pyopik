# pyopik

**pyopik** is a Python implementation of the statistical algorithm for calculating the intrinsic collision probability ($P_i$) between celestial bodies. 

It implements the formalism described by **[Dell'Oro & Paolicchi (1998)](https://ui.adsabs.harvard.edu/abs/1998Icar..136..328D/abstract)**, utilizing a Monte Carlo integration of the Kessler/Öpik theory. This approach is numerically stable for high eccentricities and inclinations, overcoming limitations of analytical approximations[cite: 37, 130].

## Features
* **Accurate Geometry:** Uses the exact Jacobian transformation ($|J|$) from orbital elements to Cartesian space[cite: 493].
* **Monte Carlo Integration:** Handles residence time weighting ($\Delta$) correctly for Keplerian orbits[cite: 122].
* **Velocity Calculation:** Returns both the intrinsic probability ($P_i$) and the mean impact velocity ($U$).

## Installation

You can install this package locally using pip:

```bash
git clone [https://github.com/bumhooLIM/pyopik.git](https://github.com/bumhooLIM/pyopik.git)
cd pyopik
pip install .