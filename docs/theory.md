# Theoretical Background

## 1. Finite Element Formulation

### 1.1 Element Stiffness Matrix
For a 2D truss element connecting nodes $i$ and $j$:

$$
\mathbf{k}^{(e)} = \frac{EA}{L}
\begin{bmatrix}
c^2 & cs & -c^2 & -cs \\
cs & s^2 & -cs & -s^2 \\
-c^2 & -cs & c^2 & cs \\
-cs & -s^2 & cs & s^2
\end{bmatrix}
$$

where $c = \cos\theta$, $s = \sin\theta$, $L$ = element length, $E$ = Young's modulus, $A$ = cross-sectional area.

### 1.2 Thermal Loading
$$\Delta L_T = \alpha \cdot \Delta T \cdot L$$

### 1.3 Generalized Clapeyron Theorem
$$W_{mech} = U_{strain} + \frac{1}{2} W_{prestress}$$

## 2. Boundary Conditions
- **Elimination** (default): Remove fixed DOFs
- **Penalty**: Add large stiffness to constrained DOFs

## 3. Buckling Analysis
Euler critical load:
$$P_{cr} = \frac{\pi^2 E I}{(K L)^2}$$
