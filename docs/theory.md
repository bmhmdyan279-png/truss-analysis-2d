# Theoretical Background

This document is the reference for *why* the code does what it does. Each
section states the governing relation, the assumption behind it, and — where an
assumption is a modelling choice rather than a physical law — what breaks if it
is violated.

---

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

where $c = \cos\theta$, $s = \sin\theta$, $L$ = element length, $E$ = Young's
modulus, $A$ = cross-sectional area.

This matrix is **rank one**. Writing the compatibility (elongation) vector

$$
\mathbf{b}^{(e)} = [-c,\ -s,\ c,\ s]^T ,
$$

the element matrix is exactly the dyad

$$
\mathbf{k}^{(e)} = k_e\, \mathbf{b}^{(e)} \mathbf{b}^{(e)T}, \qquad k_e = \frac{EA}{L}.
$$

That single identity is what the whole library is built on. It gives the global
assembly as a sum of rank-one terms,

$$
\mathbf{K} = \sum_e k_e\, \mathbf{b}_e \mathbf{b}_e^T = \mathbf{B}^T \operatorname{diag}(\mathbf{k})\, \mathbf{B},
$$

with $\mathbf{B}$ the compatibility matrix whose rows are the $\mathbf{b}_e$.
Two consequences are used directly:

* **Vectorised assembly.** There is no need for a loop over the $4\times4$
  block; the row indices, column indices and values for the whole model are
  three arrays and `coo_matrix` performs the duplicate-index summation.
* **Exact rank-one perturbation.** Softening one member changes $\mathbf{K}$ by
  a single dyad, which the Sherman–Morrison formula inverts in closed form
  (section 6). This is why every member's criticality can be obtained from one
  factorisation instead of one per member.

Both identities are asserted in `tests/test_cross_module_consistency.py`
against the assembled matrix, independently of the assembler's own code path.

### 1.2 Model scope — what is *not* included

A pin-jointed truss member carries axial force alone, so the solver uses
$k_e = EA/L$ and nothing else. It is worth being explicit about what that
excludes, because several element fields exist for post-processing rather than
for the stiffness matrix:

| Field | Enters $\mathbf{K}$? | Used by |
|---|---|---|
| `E`, `A` | **yes** | stiffness |
| `alpha`, `delta_T`, `delta_L_free` | no — they form the load vector | imposed strain |
| `I_sec`, `effective_length_factor` | no | buckling checks only |
| `density` | no | optional self-weight |

Consequences that follow from the formulation, not from an implementation
shortcut:

* **No bending.** $I_{sec}$ never affects a displacement. Supplying it changes
  only the buckling report.
* **No geometric nonlinearity.** There is no P-$\Delta$ or large-displacement
  term. For a slender member near its buckling load, or a hot member whose
  displacements have grown because $k_E(T)$ has fallen, the linear result
  understates the true displacement. The reported buckling utilisation is an
  elastic critical load, not a nonlinear collapse load.
* **No material nonlinearity in the solve.** The five-branch EN 1993-1-2
  stress–strain law is implemented in `material/steel_eurocode.py` and used for
  capacity checks, but $\mathbf{K}$ is built from the linear modulus $k_E(T) E$.
  A member that has yielded is still modelled as linear-elastic.
* **Pinned connections only.** No rotational spring, so no semi-rigid behaviour.
* **Nodal loads only.** Member self-weight is lumped to the two end nodes when
  `density` is supplied; there is no distributed load.

### 1.3 Thermal and fabrication loading

An imposed elongation

$$
\Delta L_{prestress} = \alpha \cdot \Delta T \cdot L + \Delta L_{free}
$$

produces no stress by itself — it is a *strain*, not a load. In the displacement
method it enters through the equivalent nodal force pair

$$
\mathbf{F}_{imposed} = k_e\, \Delta L_{prestress}\, \mathbf{b}_e ,
$$

which is self-equilibrated along the member axis. The total elongation is the
part from nodal displacements,

$$
\Delta L_{total} = \mathbf{b}_e^T \mathbf{U}, \qquad
\Delta L_{mech} = \Delta L_{total} - \Delta L_{prestress},
$$

and **only the mechanical part produces force**: $N = k_e\, \Delta L_{mech}$.

Two limits are worth stating because they are the ones most often computed
wrongly:

* *Free expansion* (determinate structure): the nodes move by
  $\Delta L_{prestress}$, so $\Delta L_{mech} = 0$, $N = 0$ and the stored strain
  energy is **zero** even though the member is visibly longer.
* *Fully restrained*: the nodes do not move, $\Delta L_{total} = 0$, so
  $\Delta L_{mech} = -\Delta L_{prestress}$ and $N = -EA\alpha\Delta T$. The
  member is fully stressed while every displacement is exactly zero.

Both are asserted against closed forms in `tests/test_property_invariance.py`
and `tests/test_self_equilibrated_energy.py`.

The assembler keeps three vectors distinct, and the distinction is load-bearing
rather than cosmetic:

$$
\underbrace{\mathbf{F}_{mechanical}}_{\text{applied loads}} \quad
\underbrace{\mathbf{F}_{imposed}}_{\text{thermal / fabrication}} \quad
\mathbf{F}_{ext} = \mathbf{F}_{mechanical} + \mathbf{F}_{imposed}
$$

The energy check (next section) needs $\mathbf{F}_{mechanical}$ **alone**; using
$\mathbf{F}_{ext}$ there would silently double-count the imposed strain.

### 1.4 Generalized Clapeyron Theorem

With $\Delta L = \Delta L_{mech} + \Delta L_{prestress}$, and defining

$$
U_{strain} = \tfrac12 \sum_e k_e \Delta L_{mech,e}^2, \qquad
W_{prestress} = \sum_e k_e\, \Delta L_{prestress,e}\, \Delta L_{mech,e}, \qquad
W_{mech} = \tfrac12 \mathbf{U}^T \mathbf{F}_{mechanical},
$$

a correct linear-elastic solve satisfies

$$
\boxed{\,W_{mech} = U_{strain} + \tfrac12 W_{prestress} + U_{penalty}\,}
$$

Derivation:

```text
K U = F_mechanical + F_imposed
U^T K U = U^T F_mechanical + U^T F_imposed

U^T K U = sum(k * delta_L^2)
        = 2 U_strain + 2 W_prestress + sum(k * delta_L_prestress^2)

U^T F_imposed = sum(k * delta_L_prestress * delta_L)
              = W_prestress + sum(k * delta_L_prestress^2)

=>  2 U_strain + W_prestress = U^T F_mechanical
=>  0.5 * U^T F_mechanical   = U_strain + 0.5 * W_prestress
```

**The self-equilibrated case is not special.** Setting $\mathbf{F}_{mechanical} = 0$
gives $U_{strain} = -\tfrac12 W_{prestress}$, which is generally *non-zero*. A
fully restrained heated bar stores a large strain energy while doing no external
work at all. An earlier revision of `check_energy` short-circuited this branch
and demanded $U_{strain} \approx 0$, which made the textbook thermal-stress
problem raise `EnergyValidationError`. It is now tested in both limits.

**$U_{penalty}$** is zero under the default elimination method. Under penalty
boundary conditions the constraint springs are real springs and store
$\tfrac12 \alpha \sum_{d} U_d^2$. Omitting that term leaves a residual of order
$R^2/\alpha$ — about 0.6 % at $\alpha = 10^{12}$ on the shipped example. The
constrained DOFs must **not** be zeroed before the check: at that penalty a
support settles $\sim 10^{-7}$ m while free nodes move $\sim 10^{-6}$ m, so a
zeroed field is not the solution of any system.

### 1.5 Scaling of numerical tests

Every threshold in the library is relative and dimensionless wherever physics
allows, because an absolute joule, newton or metre cut-off behaves differently
for a millimetre lattice and a kilometre bridge. The policy lives in one place
(`truss_analysis.numerics.NumericalTolerances`) rather than being scattered:

| Quantity | Scale used |
|---|---|
| Energy residual | largest of $\|W_{mech}\|$, $\|U_{strain}\|$, $\tfrac12\|W_{prestress}\|$, $\|U_{penalty}\|$; floored by $\varepsilon \cdot \sum \tfrac12 k \Delta L_{prestress}^2$ |
| Force sign classification | axial strain $N/(EA)$ against $10^{-12}$ |
| Equilibrium forces | $\sum |F_x| + |F_y|$ over loads and reactions |
| Equilibrium moments | that force scale $\times$ bounding-box diagonal |

The last line matters: force and moment residuals have different dimensions,
so they cannot share a tolerance. An earlier revision scaled the moment bound by
the force reference *twice*, giving a bound in units of force squared — far too
tight for a large bridge and far too loose for a small one.

The energy floor exists because imposed-strain problems can make **every** term
of the balance vanish analytically (free expansion), where a purely relative
test divides round-off by round-off and reports a 100 % error on an exact
result. Anchoring the floor to the characteristic imposed-strain energy keeps it
unit-agnostic; a hard-coded joule constant does not.

---

## 2. Boundary Conditions

### 2.1 Elimination (default)

The constrained DOFs are removed and the reduced system $\mathbf{K}_{ff}\mathbf{U}_f = \mathbf{F}_f$
is solved. The constraints are satisfied **exactly**, the system is smaller, and
no artificial stiffness is introduced. This is the right choice almost always.

### 2.2 Penalty

A large term $\alpha$ is added to each constrained diagonal and the full system
is solved:

$$
(\mathbf{K} + \alpha \mathbf{P})\,\mathbf{U} = \mathbf{F},
\qquad U_d \approx \frac{F_d}{K_{dd} + \alpha} \to 0
$$

$\alpha$ may be supplied as an absolute stiffness in N/m (`penalty_value`, the
conventional meaning) or derived as `penalty_multiplier` $\times \max|\operatorname{diag}\mathbf{K}|$,
which is dimensionless and therefore safe across unit systems.

**The trade-off is exact and unavoidable.** Measured on the shipped example
($\operatorname{cond}(\mathbf{K}_{ff}) \approx 41$):

$$
\text{relative constraint error} \approx \frac{0.4}{\alpha / \max|\operatorname{diag} K|},
\qquad
\operatorname{cond}(\mathbf{K}_{pen}) \approx 24 \cdot \frac{\alpha}{\max|\operatorname{diag} K|}
$$

so their **product is constant** ($\approx 9.4$) regardless of $\alpha$. There is
no free lunch, only a choice of where to sit:

| ratio $\alpha/\max|\operatorname{diag}K|$ | constraint error | $\operatorname{cond}(\mathbf{K}_{pen})$ |
|---|---|---|
| $10^{4}$ | $4\times10^{-5}$ | $2\times10^{5}$ |
| $10^{8}$ | $4\times10^{-9}$ | $2\times10^{9}$ |
| $10^{10}$ (default) | $4\times10^{-11}$ | $2\times10^{11}$ |
| $10^{12}$ | $4\times10^{-13}$ | $2\times10^{13}$ — beyond double precision |

Ratios outside $[10^{4}, 10^{12}]$ emit `IllConditionedWarning` **in both
directions**. The lower bound matters more often than the upper one: an absolute
`penalty_value` chosen for one model can be tens of times the diagonal of
another. The shipped examples' `penalty_value = 1e12` is only $\approx 56\times$
their own stiffness diagonal, giving a 0.3 % constraint error — which the
warning now reports instead of leaving the user to discover it as a discrepancy
against the elimination solve.

---

## 3. Support semantics

A degree of freedom is constrained if and only if the node is flagged
`is_support` **and** the matching `support_dx` / `support_dy` restraint is set.
Neither alone is sufficient. Because a restraint without the flag is silently
ignored — producing a *different structure* from the one described — both
inconsistent combinations are rejected by `validate_inputs`.

The node $\to$ constrained-DOF map is defined once, in
`model.fixed_dof_indices`, and consumed by both the assembler and the
criticality engine. They previously carried private copies, which could
disagree in a way no equivalence test on $\mathbf{K}$ would catch, because both
paths would be wrong together.

### 3.1 Determinacy, stability and conditioning are three different things

$$
\underbrace{m + r - 2j}_{\text{static determinacy}} \quad\ne\quad
\underbrace{\operatorname{rank}(\mathbf{K}_{ff})}_{\text{kinematic stability}} \quad\ne\quad
\underbrace{\operatorname{cond}(\mathbf{K}_{ff})}_{\text{numerical quality}}
$$

* **Determinacy** is an algebraic *count*. It is necessary but not sufficient
  for stability: three parallel rollers in a line satisfy $r \ge 3$ and are
  still a mechanism.
* **Stability** is a numerical fact about $\mathbf{K}_{ff}$, decided by its rank.
* **Conditioning** is a statement about round-off amplification. It is *not*
  validity: an ill-conditioned matrix still has a unique solution.

`validate_inputs`'s "at least 3 constraints" check is documented as a **sanity
check**, not a stability proof, for exactly this reason.
`graph_validation.TopologyReport` reports all three separately, and
`numerics.NumericalStatus` distinguishes `stable` / `ill_conditioned` /
`singular` so a result can carry its numerical evidence rather than only its
value.

Two rank cutoffs are used deliberately and come from one policy:

| Cutoff | Value | Role |
|---|---|---|
| `rank_rel_cutoff` | $10^{-13}$ | **hard gate** in the solver. Refusing a solvable structure is worse than solving a marginal one. |
| `near_singular_rel_cutoff` | $10^{-9}$ | **diagnostic** in validation. Flagging a near-mechanism as suspect is cheap and informative. |

Previously these two numbers lived in different files and had drifted apart, so
a model could be reported as a mechanism by the validator and as perfectly
solvable by the solver.

---

## 4. Buckling

### 4.1 Three distinct questions

"Buckling" is used loosely for three different things, and conflating them is
the main source of error in this area:

1. **Euler elastic critical load** — a property of the member's stiffness:
   $$N_{cr} = \frac{\pi^2 E I}{(K L)^2}$$
2. **Member buckling resistance** — a *design capacity*, reduced below $N_{cr}$
   by imperfections (section 4.2).
3. **Structural stability** — an eigenvalue problem for the whole system, which
   this library does not solve; it is detected instead as rank deficiency of
   $\mathbf{K}_{ff}$.

$K$ is the effective-length factor. It is honoured in `calculate_buckling`,
`limitstates` and `sections.euler_buckling_load` — all three now delegate to the
same function, having previously carried three separate copies of the formula,
one of which ignored $K$.

A member carrying compression with no usable `I_sec` is reported as
`status="unknown"` with `safe=False` and a `BucklingCheckWarning`. It previously
returned `safe=True`, i.e. **missing data was interpreted as safety**.

### 4.2 EN 1993-1-2 §4.2.3.1 — fire buckling resistance

The elastic critical load alone is not a capacity. At intermediate slenderness
it overestimates what the member can carry, because it ignores residual stresses
from rolling and initial out-of-straightness. The code applies a reduction factor
$\chi$ from a buckling curve:

$$
\bar\lambda_\theta = \sqrt{\frac{A\, f_{y,\theta}}{N_{cr}}}, \qquad
f_{y,\theta} = k_{y,\theta} f_y, \qquad
E_\theta = k_{E,\theta} E
$$

$$
\Phi_\theta = \tfrac12\left[1 + \alpha\,(\bar\lambda_\theta - 0.2) + \bar\lambda_\theta^{\,2}\right],
\qquad
\chi = \frac{1}{\Phi_\theta + \sqrt{\Phi_\theta^2 - \bar\lambda_\theta^{\,2}}} \le 1
$$

$$
\boxed{\,N_{b,fi,\theta,Rd} = \frac{\chi\, A\, f_{y,\theta}}{\gamma_{M,fi}} = \chi \cdot N_{Rd}\,}
$$

with $\alpha = 0.65\,\alpha_c$ and $\alpha_c$ from the EN 1993-1-1 Table 6.1/6.2
curve (default `c`, $\alpha_c = 0.49$), per §4.2.3.1(3).

This one expression covers **both** limits, which is why it replaces a
`min(N_cr, N_Rd)` approximation:

* $\bar\lambda_\theta \to 0 \Rightarrow \chi \to 1$: the capacity is the yield
  resistance (stocky member, buckling need not be checked below
  $\bar\lambda = 0.2$ per EN 1993-1-1 §6.3.1(4)).
* $\bar\lambda_\theta \to \infty \Rightarrow \chi \to 1/\bar\lambda_\theta^2$: the
  capacity tends to the Euler load.

**Temperature makes members slender.** Because $k_E$ falls faster than $k_y$,

$$
\bar\lambda_\theta = \sqrt{\frac{k_{y,\theta}}{k_{E,\theta}}}\;\bar\lambda_{20^\circ C}
$$

*grows* with temperature. A member comfortably stocky at 20 °C can become
buckling-governed at 600 °C. The `min()` form could not express this at all,
since it compared two numbers rather than evaluating one curve.

Measured effect on the validation case (two-bar truss, 600 °C,
$\bar\lambda_\theta = 0.655$): $\chi = 0.8175$, capacity 902 961 N instead of
1 104 500 N, DCR 0.0831 instead of 0.0679 — **22 % more conservative**. The
legacy model remains selectable via `BucklingModel.EULER_ONLY` so previously
published numbers stay reproducible, and a test asserts $\chi$ never yields a
larger capacity than `min(N_cr, N_Rd)`, so the reduction cannot be wired
backwards without failing.

The implementation reproduces the published curve-c value $\chi = 0.54$ at
$\bar\lambda = 1.0$ exactly.

---

## 5. Criticality Index — definition and physical meaning

### 5.1 What the index measures

The criticality index answers a counterfactual question:

> *If this member lost part of its stiffness, how much worse would the
> structure's response become?*

Formally, with $\alpha \in (0, 1]$ the retained stiffness fraction and
$\mathbf{K}(\alpha)$ the structure with member $i$ scaled by $\alpha$:

$$
CI_i(\alpha) = \frac{\max_j |U_j(\alpha)|}{\max_j |U_j(1)|} - 1
$$

So $CI_i$ is the **fractional increase in the peak nodal displacement** when
member $i$ is damaged. It is dimensionless, zero for a member whose removal
changes nothing, and unbounded above.

### 5.2 Why it is a *relative* measure, and against which baseline

The baseline is the **undamaged structure at the same temperature field**. This
matters and is easy to misread. In `ci_two_component` the temperature first
scales every $E_i$ by $k_E(T)$, and *then* the perturbation is applied on top.
The reported index therefore describes

```text
baseline  ->  thermal degradation  ->  member perturbation  ->  damaged state
                (already applied)        (what CI measures)
```

and is **not** a damage index for the fire itself. Reading $CI_i$ as "how much
of the fire damage is due to member $i$" is wrong; it is "how much additional
degradation occurs if member $i$ is further weakened, given the fire state
already in place".

### 5.3 Why the displacement definition is not enough

$CI_i$ is a scalar summary of a vector field, and summaries lose information.
Two members can be equally damaging in different ways:

```text
member A: max|U| rises 20%, forces barely redistribute
member B: max|U| rises  2%, a neighbouring member's force rises 80%,
          total strain energy rises 30%
```

$CI_u$ ranks A far above B, even though B is the member whose loss drives
another member past its capacity. In a fire-triage or retrofit study — where
what fails a structure is usually a *force* limit state — that is the wrong
answer.

The library therefore reports five indices from the same exact perturbed field
(`criticality/criteria.py`):

| Index | Definition | Question it answers |
|---|---|---|
| $CI_u$ | $\max|U_{pert}| / \max|u| - 1$ | Does the structure deform much more? |
| $CI_{N,max}$ | $\max|N_{pert}| / \max|N_{base}| - 1$ | Does *any* member get overloaded? |
| $CI_{N,self}$ | $|N_{pert}[i]| / |N_{base}[i]| - 1$ | Was this member carrying load at all? |
| $CI_E$ | $E_{pert} / E_{base} - 1$ | Does the structure absorb more energy? |
| $CI_R$ | $\max|R_{pert}| / \max|R_{base}| - 1$ | Do the supports see more load? |

$CI_{N,self}$ has a useful interpretation on its own: a member with
$CI_{N,self} \approx 0$ is one whose force is insensitive to its own stiffness,
i.e. a member the structure does not really need. That is a redundancy measure
rather than a criticality measure, and the two are complementary.

`governing` names the largest index; `composite` is the maximum, which never
understates an individual effect and is the right default for triage.

### 5.4 Temperature invariance of the ranking

Under a **uniform** temperature field every stiffness is scaled by the same
factor: $\mathbf{K}(\theta) = k_E(\theta)\,\mathbf{K}_0$. With unchanged loads,

$$
\mathbf{U}(\theta) = \frac{\mathbf{U}_0}{k_E(\theta)}, \qquad
\mathbf{N}(\theta) = \mathbf{N}_0
$$

so displacements scale uniformly and **forces do not change at all**. Every
$CI_i$ is a *ratio* of two states at the same temperature, so the common factor
cancels and the ranking is invariant: Kendall $\tau_b = 1$ against the ambient
baseline.

This is a mathematical property of the uniform case, and it is worth being
honest about its limits: real fire fields are never exactly uniform, and under a
gradient $k_E$ differs per member, the factor does not cancel, and the ranking
can and does change. The invariance test in `test_uniform_invariance.py` is
therefore a *regression guard on the machinery*, not evidence that criticality
is temperature-independent in general. It is validated with "antifake"
providers — constant or drifting fake $k_E$ functions must be **rejected** by
the same test the real one passes.

### 5.5 Validity limit (explicit)

The rank-one path is exact **only for single-member perturbations**. For
simultaneous multi-member perturbations use `perturb_multi` (Woodbury rank-$r$)
or a full re-solve. Applying the rank-one formula member-by-member to a
multi-member change is wrong, and no public function in the package does it.

---

## 6. Rank-1 (Sherman–Morrison) Perturbation Engine

Reducing the axial stiffness of a single truss member is a rank-one update,
because $\mathbf{k}_i = k_i \mathbf{b}_i \mathbf{b}_i^T$:

$$
\mathbf{K}_{pert} = \mathbf{K} + \Delta_i\, \mathbf{b}_i \mathbf{b}_i^T,
\qquad \Delta_i = (\alpha - 1)\, k_i
$$

$$
(\mathbf{K} + \Delta \mathbf{b}\mathbf{b}^T)^{-1}\mathbf{F}
= \mathbf{u} - \frac{\Delta\, (\mathbf{K}^{-1}\mathbf{b})\,(\mathbf{b}^T \mathbf{u})}{1 + \Delta\, \mathbf{b}^T \mathbf{K}^{-1}\mathbf{b}}
$$

Factorising $\mathbf{K}_{ff}$ **once** and solving for the whole compatibility
matrix $\mathbf{B}$ (giving $\mathbf{Z} = \mathbf{K}^{-1}\mathbf{B}$) yields every
member's perturbed state in one vectorised pass:

$$
d_i = \mathbf{b}_i^T \mathbf{Z}_i, \quad
f_i = \mathbf{b}_i^T \mathbf{u}, \quad
\text{coef}_i = \frac{\Delta_i f_i}{1 + \Delta_i d_i}, \quad
\mathbf{U}_{pert} = \mathbf{u} - \mathbf{Z}\,\operatorname{diag}(\text{coef})
$$

Column $i$ of $\mathbf{U}_{pert}$ is the complete perturbed state for member $i$.
The cost is one factorisation plus $O(n_E \cdot n_{dof})$ work, instead of one
factorisation per member.

**Numerical guard.** $1 + \Delta_i d_i \to 0$ means the perturbed structure is
(near) a mechanism. Such members are flagged and routed to a full brute-force
solve; if that solve is singular the member's CI is $+\infty$ with a `mechanism`
flag. The engine never emits a silent finite number for a guarded member, and
the multi-criteria indices propagate $+\infty$ rather than computing from a
zeroed placeholder.

**Perturbed forces need the perturbed stiffness.** Since
$N = \alpha k_i e_i$ for the softened member, its elongation grows to
$N/(\alpha k_i)$. Multiplying that back by the *unperturbed* $k_i$ returns
$N/\alpha$ — at $\alpha = 0.5$ exactly a factor of two too large. This shows up
as a spurious doubling of the force index even in a statically **determinate**
truss, where softening a member cannot change any force at all. That determinate
case is asserted precisely because it catches the error.

**Reactions need a rank-one correction.** With
$\mathbf{R} = (\mathbf{K}\mathbf{U})[\text{fixed}]$ and the constrained DOFs zero,

$$
\mathbf{R}_{pert} = \mathbf{K}[\text{fixed}, \text{free}]\,\mathbf{u}_{pert}
+ \Delta_i\, \mathbf{b}_i[\text{fixed}]\,(\mathbf{b}_i[\text{free}]^T \mathbf{u}_{pert})
$$

The second term is the perturbed member's own contribution to the supports and
must not be dropped.

---

## 7. Adjoint Sensitivity (DDM)

`sensitivity.py` provides an independent cross-check on member importance via
the Direct Differentiation Method. With $\mathbf{Z} = \mathbf{K}_{ff}^{-1}\mathbf{B}$
and $dk_i/dA_i = E_i / L_i$:

$$
\frac{d\mathbf{U}}{dA_i} = -\frac{E_i}{L_i}\, \mathbf{Z}_i\, (\mathbf{b}_i^T \mathbf{U}_f)
$$

One factorisation serves all members, and no explicit inverse is formed. The
reported member strain energy is the **mechanical** one,
$\tfrac12 k\, \Delta L_{mech}^2$, consistent with section 1.3 — the quadratic
form $\tfrac12 \mathbf{u}_e^T \mathbf{k}_e \mathbf{u}_e$ equals
$\tfrac12 k \Delta L_{total}^2$ and therefore *includes* imposed strain. Both
routes are computed and retained (`strain_energy` and `strain_energy_total`);
they coincide exactly when no imposed strain is present, which is what keeps the
cross-check meaningful.

---

## 8. Verification Matrix

Which claim is supported by which kind of evidence. "Independent" means the
check shares no factorisation, assembly path or constant with the code it
verifies.

| Feature | Analytical | Independent solver | Property / invariance | Brute force | Thermal |
|---|:--:|:--:|:--:|:--:|:--:|
| $\mathbf{K}$ assembly | ✓ | — | ✓ (sym, PSD, rank-1) | — | ✓ |
| $\mathbf{K} = \mathbf{B}^T \operatorname{diag}(k)\mathbf{B}$ | ✓ | — | ✓ | — | — |
| Sparse vs dense assembly | — | ✓ | ✓ | — | ✓ |
| Displacement field | ✓ | ✓ | ✓ (linearity, superposition) | ✓ | ✓ |
| Reactions | ✓ | ✓ | ✓ | ✓ | — |
| Equilibrium residuals | — | — | ✓ (dimensional scaling) | — | ✓ |
| Energy balance | ✓ | — | ✓ (scale invariance) | — | ✓ both limits |
| Penalty BC | — | ✓ (vs elimination) | ✓ (error↔cond trade-off) | — | — |
| Rigid-body modes | ✓ | — | ✓ (subspace projectors) | — | — |
| Translation invariance | — | — | ✓ | — | — |
| Rotation invariance | — | — | ✓ (5 angles + controls) | — | — |
| Node renumbering | — | — | ✓ (3 orderings) | — | — |
| Element orientation | — | — | ✓ (full + partial swap) | — | — |
| Free thermal expansion | ✓ closed form | — | ✓ | — | ✓ |
| Restrained thermal stress | ✓ closed form | — | ✓ | — | ✓ |
| Fabrication misfit | ✓ closed form | — | ✓ (determinate = no force) | — | — |
| Euler $N_{cr}$ | ✓ | — | — | — | ✓ via $k_E$ |
| $\chi$ buckling curve | ✓ (published $\chi = 0.54$ at $\bar\lambda=1$) | — | ✓ (monotone, bounded) | — | ✓ |
| $k_E, k_y, k_p$ tables | ✓ (fixture) | — | ✓ (clamping) | — | ✓ |
| Stress–strain law | ✓ | — | ✓ (invertibility) | — | ✓ |
| Rank-1 CI | — | ✓ (full re-solve) | ✓ | ✓ (equivalence) | ✓ |
| Woodbury rank-$r$ | — | ✓ | — | ✓ | ✓ |
| Multi-criteria CI | — | ✓ (all 5 indices) | ✓ (determinate ⇒ $CI_N = 0$) | ✓ | ✓ |
| Uniform-$T$ ranking invariance | ✓ (proved) | — | ✓ + antifake providers | — | ✓ |
| OpenSeesPy cross-check | — | ✓ (optional extra) | — | — | ✓ |

Gaps that are honestly open: no experimental benchmark (e.g. the Cardington
fire tests), no FORM/SORM for small failure probabilities, no nonlinear
geometric or material solve, and the OpenSeesPy bridge is an optional extra so
public CI does not run it.
