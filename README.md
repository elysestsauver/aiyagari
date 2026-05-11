# Aiyagari (1994) Model — Value Function Iteration

A Python implementation of the Aiyagari (1994) heterogeneous-agent incomplete-markets model, adapted from MATLAB code by [Oliko Vardishvili](https://github.com/ovardish/2022_Computational_Course_Code).

Households face uninsurable idiosyncratic income shocks, save in a single risk-free asset subject to a borrowing constraint, and the equilibrium interest rate is determined by aggregate asset supply equalling capital demand from a representative firm.

> Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *Quarterly Journal of Economics* 109(3): 659–684.

## Algorithm

1. **Income process** — discretize an AR(1) productivity process using the Rouwenhorst approximation (via `quantecon`)
2. **Household problem** — for a given interest rate *r*, solve for the optimal savings policy via value function iteration on a finite asset grid
3. **Stationary distribution** — iterate the joint distribution over (assets, productivity) until convergence
4. **General equilibrium** — find the equilibrium *r* by bisection: stop when aggregate asset supply equals capital demand from a Cobb-Douglas firm

## Parameters

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Risk aversion | γ | 3 | CRRA utility curvature |
| Discount factor | β | 0.96 | |
| Depreciation | δ | 0.08 | |
| Capital share | α | 0.36 | Cobb-Douglas exponent |
| Borrowing limit | b | −2 | Exogenous floor on assets |
| Productivity states | n_z | 7 | Number of Markov states |
| AR(1) persistence | ρ | 0.2 | |
| Long-run std. dev. | σ_LR | 0.4 | Of log income |

## Outputs

Running the script produces three plots:

- **Asset policy function** — next-period assets *k'* as a function of current assets *k*, for each productivity state
- **Consumption policy function** — consumption *c* as a function of *k*, for each productivity state
- **Stationary distribution** — marginal distribution of assets across households

## Requirements

```
pip install numpy quantecon matplotlib
```

## Usage

```bash
python aiyagari.py
```

Equilibrium progress is printed at each bisection step:

```
k0 = 4.3021, k1 = 5.1847, r0 = 0.0208, r1 = 0.0183
k0 = 4.3021, k1 = 4.7231, r0 = 0.0104, r1 = 0.0164
...
```
