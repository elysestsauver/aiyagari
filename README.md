# Aiyagari

A Python implementation of the Aiyagari (1994) heterogeneous-agent
incomplete-markets model. Households face uninsurable idiosyncratic income
shocks, save in a single risk-free asset subject to a borrowing constraint,
and the equilibrium interest rate is determined by aggregate asset supply
equaling capital demand from a representative firm.

> Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate
> Saving." *Quarterly Journal of Economics* 109(3): 659–684.

## Layout

```
.
├── src/aiyagari/
│   ├── income.py          # Markov approximations of AR(1): Tauchen, Rouwenhorst
│   ├── household.py       # Household problem (value-function iteration)
│   ├── distribution.py    # Stationary distribution over (a, z)
│   └── equilibrium.py     # General-equilibrium loop for r
├── tests/                 # pytest smoke tests
├── notebooks/             # Jupyter notebooks for exploration
└── data/                  # Empty; for any external data you bring in
```

## Setup

### Option A: conda

```bash
conda env create -f environment.yml
conda activate aiyagari
pip install -e .
```

### Option B: pip + venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

After either, verify:

```bash
pytest                      # should pass
jupyter lab notebooks/      # to open the walkthrough notebook
```

## Quick start

```python
from aiyagari.income import rouwenhorst
from aiyagari.household import solve_household
from aiyagari.distribution import stationary_distribution
from aiyagari.equilibrium import find_equilibrium_r

# 1. Discretize an AR(1) income process
z_grid, P = rouwenhorst(n=7, rho=0.9, sigma=0.2)

# 2. Solve the household problem at a candidate r
V, policy = solve_household(z_grid, P, r=0.03, beta=0.96, gamma=2.0)

# 3. Find the stationary distribution
mu = stationary_distribution(policy, P)

# 4. Or jump straight to general equilibrium
r_star, K_star = find_equilibrium_r(z_grid, P, beta=0.96, gamma=2.0, alpha=0.36, delta=0.08)
```

The walkthrough notebook in `notebooks/01_aiyagari_walkthrough.ipynb`
runs all of the above end-to-end with default parameters and produces the
canonical asset-supply / capital-demand plot.

## Status

Initial scaffolding. The Markov approximation routines are complete; the
household solver uses standard value-function iteration on a fixed grid
(slow but reliable). EGM and Reiter linearization are good next steps for
speed.
