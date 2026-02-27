# Illustration Doc on $V(r)$ Estimation

### Purpose

This note explains how to **compute / handle** the adjustment vector $V(r)$ in simulations that verify the theory in `theory/theory.md`, and how to do **proxy-only recovery** (using only $(\tilde R,\hat Z)$ + known noise mechanism).

### Key definitions (from theory)

Residuals:
$$
R = Y - h(Z),\qquad \tilde R = Y - h(\hat Z),\qquad \delta = h(1)-h(0).
$$
Target conditional residual distributions (pmf/density):
$$
f_z(r) = P(R=r \mid Z=z),\quad z\in\{0,1\}.
$$

Known joint label probabilities (from $\pi$ and $P(\hat Z\mid Z)$):
$$
p_{ab}=P(\hat Z=a, Z=b),\quad a,b\in\{0,1\},\qquad
M=\begin{pmatrix}p_{11}&p_{10}\\p_{01}&p_{00}\end{pmatrix}.
$$

Forward relations:
$$
\begin{aligned}
g_1(r)&:=P(\tilde R=r,\hat Z=1)=p_{11}f_1(r)+p_{10}f_0(r+\delta),\\
g_0(r)&:=P(\tilde R=r,\hat Z=0)=p_{01}f_1(r-\delta)+p_{00}f_0(r).
\end{aligned}
$$

Adjustment vector:
$$
V(r)=
\begin{pmatrix}
p_{10}\big(f_0(r+\delta)-f_0(r)\big)\\
p_{01}\big(f_1(r-\delta)-f_1(r)\big)
\end{pmatrix}.
$$

Inversion (Proposition):
$$
\begin{pmatrix} f_1(r) \\ f_0(r) \end{pmatrix}
= M^{-1}\left[
\begin{pmatrix} g_1(r) \\ g_0(r) \end{pmatrix}
- V(r)
\right].
$$

**Important:** $V(r)$ depends on unknown $f$. In proxy-only settings, do not treat $V(r)$ as directly estimable.

---

### What to estimate from data
From simulated/calibration data (where only $(\tilde R_i,\hat Z_i)$ are “observed” in proxy-only mode), estimate $g_1(r)$ and $g_0(r)$ on a grid via histogram/KDE.
- If you first estimate conditional $P(\tilde R=r\mid \hat Z=z)$, convert to joint:
    $$
    g_z(r) = P(\tilde R=r\mid \hat Z=z)\cdot P(\hat Z=z).
    $$

---

***Two simulation tracks that need to be implemented:***

### 1. Oracle
Goal: verify the proposition and the sign/direction of shifts.

1. Use true $Z$ to compute $R_i = Y_i-h(Z_i)$.
2. Estimate $f_0,f_1$ from $(R,Z)$ (empirical pmf / KDE).
3. Compute $V_{\text{oracle}}(r)$ by definition (using oracle $f$).
4. Plug $(g,V_{\text{oracle}})$ into the proposition and verify recovered $f$ matches oracle $f$ (up to sampling error).

This isolates “math correctness” from numerical inversion issues.

### 2. Proxy
Goal: recover $f_0,f_1$ using only $(\tilde R,\hat Z)$ + known $(p_{ab})$.

Because $f(r\pm\delta)$ appears, solve for the entire vectors $\mathbf f_0,\mathbf f_1$ at once.

1. Choose residual grid $r_1,\dots,r_K$ covering main mass (expand the grid so $r\pm\delta$ mostly stays inside).
2. Build shift/interpolation matrices:
   - $S^{(+\delta)}\mathbf f_0 \approx (f_0(r_1+\delta),\dots,f_0(r_K+\delta))^\top$
   - $S^{(-\delta)}\mathbf f_1 \approx (f_1(r_1-\delta),\dots,f_1(r_K-\delta))^\top$

   Use linear interpolation when $\delta$ is not a multiple of grid spacing.
3. Stack forward equations for all grid points:
   $$
   \begin{pmatrix}\mathbf g_1\\\mathbf g_0\end{pmatrix}
   \approx
   \begin{pmatrix}
   p_{11}I & p_{10}S^{(+\delta)}\\
   p_{01}S^{(-\delta)} & p_{00}I
   \end{pmatrix}
   \begin{pmatrix}\mathbf f_1\\\mathbf f_0\end{pmatrix}.
   $$
4. Solve (typically least squares) with constraints to ensure validity:
   - Nonnegativity: $\mathbf f_0\ge 0,\ \mathbf f_1\ge 0$
   - Normalization (density grid): $\sum_i f_z(r_i)\Delta \approx 1$ for each $z$
5. After solving for $\hat f$, compute $\hat V(r_i)$ from the definition using the same shift/interpolation.

---

### Boundary handling
When evaluating $f(r\pm\delta)$: expand the grid so most evaluations are in-range.
