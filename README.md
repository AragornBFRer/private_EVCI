# EVCI


### `simulate.py`
- Generates $(Z, \hat Z, Y)$ under a known proxy mechanism and residual distributions.
- Computes true residuals $R$ and observed residuals $\tilde R$.
- **Oracle check:** uses the true $f_0, f_1$ to build $V(r)$ and verifies inversion.
- **Proxy recovery:** solves the stacked linear system using only $(\tilde R, \hat Z)$ and known label noise.

**Run**
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the simulation:
   - `python simulate.py`

**Parameters**
- `--n` sample size (default `200000`)
- `--k` grid size (default `200`)
- `--delta` class shift $h(1)-h(0)$
- `--tpr`, `--tnr` proxy accuracy

**Plots**
- `python simulate.py --plot`
- Outputs PNGs in `plots/`:
   - `perf_vs_sample_size.png`
   - `perf_vs_class_shift.png`
   - `perf_vs_proxy_accuracy.png`

The script prints L1 distances between recovered and true $f_0, f_1$ for both oracle and proxy settings.
