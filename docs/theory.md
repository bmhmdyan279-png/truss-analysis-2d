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
* **The static solve is first-order.** $\mathbf{K}$ is assembled on the
  *undeformed* geometry, so the displacements and member forces returned by
  `solver.solve` carry no P-$\Delta$ or large-displacement term. For a slender
  member near its buckling load, or a hot member whose displacements have grown
  because $k_E(T)$ has fallen, that linear result understates the true
  displacement.

  This used to be stated flatly as "no geometric nonlinearity", which stopped
  being true when §9 added the geometric stiffness $\mathbf{K}_G$ and the
  linearised bifurcation load. The distinction that matters is *where* the
  second-order term enters:

  | Quantity | Uses $\mathbf{K}_G$? | Consequence |
  |---|---|---|
  | `solver.solve` displacements, member forces | no | first-order; understates near critical |
  | `stability.linearized_buckling_load_factor` $\lambda_{cr}$ | **yes** | a real bifurcation load, but linearised about the base state |
  | `tangent_verification.exact_tangent_stiffness` | **yes** | the true tangent $\mathbf{K}_E + \mathbf{K}_G$, used to measure the gap |
  | `limitstates` member utilisation | no | an elastic code check, not a collapse load |

  What is still absent is *path following*: no Newton–Raphson load stepping, no
  arc-length continuation, no post-buckling branch. $\lambda_{cr}$ is therefore a
  tangent-stiffness criterion evaluated at the base configuration, and for a
  shallow arch or toggle whose true collapse is a snap-through limit point it
  approximates that limit with an error of $O(\theta_0^2)$. §9 quantifies the
  approximation and `stability.imperfection_sensitivity` measures how much of it
  is geometry-driven; the reported buckling utilisation remains an elastic
  critical load, not a nonlinear collapse load.
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

**The fire chain obeys the same split.** The DCR / $\theta_{sys}$ / CI /
retrofit pipeline works on a temperature *field* $T_e$ rather than on
`delta_T`, so it derives the imposed elongation as
$\Delta L_{pre,e} = \alpha_e (T_e - 20\,°\mathrm{C}) L_e + \Delta L_{free,e}$
(`engine.prestress_lengths`), solves against the total right-hand side
$\mathbf{F}_{mech} + \mathbf{B}^T\operatorname{diag}(k(T))\Delta L_{pre}$
(`engine.total_load_vector`) and reports the mechanical force
$N_e = k_e(T)(\mathbf{b}_e^T\mathbf{u} - \Delta L_{pre,e})$
(`engine.member_forces`). Before this was made explicit, that chain degraded
$E(T)$ but solved against the mechanical right-hand side only, so a heated
restrained member developed real compression in `run()` while the DCR chain
saw none of it — non-conservative exactly in the local-fire scenarios the
library targets. `tests/test_thermal_demand.py` pins the two paths
(engine vs assembler) against each other and against closed forms.

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
scales every $E_i$ by $k_E(T)$ *and* loads the structure through the
restrained-expansion equivalent forces (§1.3), and *then* the perturbation is
applied on top. The reported index therefore describes

```text
baseline  ->  thermal state          ->  member perturbation  ->  damaged state
              (degradation + demand,      (what CI measures)
               already applied)
```

and is **not** a damage index for the fire itself. Reading $CI_i$ as "how much
of the fire damage is due to member $i$" is wrong; it is "how much additional
degradation occurs if member $i$ is further weakened, given the fire state
already in place".

Two DCR ratios must be kept apart, and since 2.7 `ci_two_component` reports
both explicitly instead of collapsing them into one cold-referenced number:

$$
CI_{\text{damage}\mid T}
= \frac{DCR_{\text{pert}}(T)}{DCR_{\text{base}}(T)} - 1,
\qquad
CI_{\text{fire}}
= \frac{DCR_{\text{base}}(T)}{DCR_{\text{base}}(20^\circ C)} - 1 .
$$

$CI_{\text{damage}\mid T}$ is the force-limit-state half of the
counterfactual above: both states share the temperature field, so it is
**exactly zero at $\alpha = 1$** — an unperturbed member carries no
perturbation criticality, however hot the fire. $CI_{\text{fire}}$ is the
fire-severity of the *undamaged* member and is independent of $\alpha$.
The composite is $CI_i = \max(CI_u,\ CI_{\text{damage}\mid T})$; the fire
term never enters it. Where a single triage number that folds both effects
in is wanted (retrofit shortlists, MC ladders), the library exposes the
explicit product

$$
CI_{\text{combined}}
= (1 + CI_{\text{damage}\mid T})(1 + CI_{\text{fire}}) - 1
= \frac{DCR_{\text{pert}}(T)}{DCR_{\text{base}}(20^\circ C)} - 1 ,
$$

which is bit-for-bit the pre-2.7 cold-referenced `dcr_component`. The
pre-2.7 composite hid that product *inside* the criticality: at $\alpha=1$
and 600 °C an untouched redundant truss reported `dcr_component ≈ +2.1`
with `governing = buckling`, i.e. fire degradation masquerading as damage
criticality, contaminating `governing` and every ranking built on it.

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
factor: $\mathbf{K}(\theta) = k_E(\theta)\,\mathbf{K}_0$. **When the
right-hand side does not change with temperature** — i.e. when no member
carries an imposed strain ($\alpha \Delta T = 0$ and $\Delta L_{free} = 0$) —

$$
\mathbf{U}(\theta) = \frac{\mathbf{U}_0}{k_E(\theta)}, \qquad
\mathbf{N}(\theta) = \mathbf{N}_0
$$

so displacements scale uniformly and **forces do not change at all**. Every
$CI_i$ is a *ratio* of two states at the same temperature, so the common factor
cancels and the ranking is invariant: Kendall $\tau_b = 1$ against the ambient
baseline.

**The precondition matters, because the demand chain now carries the thermal
equivalent forces (§1.3).** With $\alpha > 0$ a uniform field also scales the
restrained-expansion right-hand side, and the base state becomes

$$
\mathbf{u}(\theta) = \frac{\mathbf{z}_m}{k_E(\theta)}
+ (\theta - \theta_0)\, \mathbf{z}_\alpha,
\qquad
N_e(\theta) = N_{e,0} + k_E(\theta)\,(\theta - \theta_0)\, k_{0,e}\,
\bigl(\mathbf{b}_e^T \mathbf{z}_\alpha - \alpha_e L_e\bigr),
$$

where $\mathbf{z}_m = \mathbf{K}_0^{-1}\mathbf{F}_{mech}$ and
$\mathbf{z}_\alpha = \mathbf{K}_0^{-1}\mathbf{B}^T(k_0\,\alpha L)$.
Two honest consequences:

* **Forces** stay invariant ($N(\theta) = N_0$) when the free-expansion
  (dilation) field is kinematically admissible — the usual pin–roller truss —
  because then $\mathbf{b}_e^T \mathbf{z}_\alpha = \alpha_e L_e$ and the
  second term vanishes. Under restrained supports (e.g. pin–pin) it does not:
  uniform heating pumps real compression into the restrained members, growing
  with $k_E(\theta)(\theta - \theta_0)$.
* **Displacement-based CIs are no longer temperature-invariant in general**,
  because $\mathbf{u}(\theta)$ mixes a $1/k_E$ term with a linear-in-$\theta$
  term and the ratio of two such states does not collapse. Absolute CI values
  drift with $\theta$ — which is physically right: the same member loss hurts
  more when the structure is also fighting thermal thrust.

The invariance test in `test_uniform_invariance.py` (campaign fixtures built
without `alpha`) is therefore a *regression guard on the machinery* under the
theorem's precondition, not evidence that criticality is temperature-independent
in general. `tests/test_thermal_demand.py` pins both sides of the refined
claim: **exact** invariance (< 1e-12 drift) with $\alpha = 0$, and a
**measured, non-zero** drift for a restrained $\alpha > 0$ structure. The
antifake providers stay: constant or drifting fake $k_E$ functions must be
**rejected** by the same test the real one passes.

This is also the precise sense in which a uniform field can reorder *nothing*
while changing *everything about magnitude*: ranking invariance under uniform
heating (where it holds) says nothing about absolute risk, which grows with
temperature through both capacity reduction and — now — thermal demand.

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
f_i = \mathbf{b}_i^T \mathbf{u} - \Delta L_{pre,i}, \quad
\text{coef}_i = \frac{\Delta_i f_i}{1 + \Delta_i d_i}, \quad
\mathbf{U}_{pert} = \mathbf{u} - \mathbf{Z}\,\operatorname{diag}(\text{coef})
$$

Column $i$ of $\mathbf{U}_{pert}$ is the complete perturbed state for member $i$.
The cost is one factorisation plus $O(n_E \cdot n_{dof})$ work, instead of one
factorisation per member.

**The numerator is the mechanical elongation, not the total one.** Softening
member $i$ scales its thermal equivalent force $k_i \Delta L_{pre,i}
\mathbf{b}_i$ by the same $\alpha$ as its stiffness, so the perturbed system
is $(\mathbf{K} + \Delta_i \mathbf{b}_i\mathbf{b}_i^T)\mathbf{u}' =
\mathbf{F} + \Delta_i \Delta L_{pre,i}\mathbf{b}_i$, and solving it gives
exactly the form above with $f_i = \mathbf{b}_i^T\mathbf{u} - \Delta L_{pre,i}
= N_i / k_i$. In words: **the rank-1 coefficient is driven by the force the
member carries, not by how much it stretched** — a stress-free hot member of a
determinate truss contributes nothing to redistribute. With $\Delta L_{pre} = 0$
the formula reduces to the classical one.

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

**Reactions need a rank-one correction — and the direct support load.** With
$\mathbf{R} = (\mathbf{K}\mathbf{U})[\text{fixed}] -
\mathbf{F}_{ext}[\text{fixed}]$ (the supports carry what the stiffness pulls
*minus* whatever is applied directly at the constrained DOFs: mechanical loads
placed on support nodes and the imposed equivalent forces),

$$
\mathbf{R}_{pert} = \mathbf{K}[\text{fixed}, \text{free}]\,\mathbf{u}_{pert}
- \mathbf{F}_{ext}[\text{fixed}]
+ \Delta_i\, \mathbf{b}_i[\text{fixed}]\,
(\mathbf{b}_i[\text{free}]^T \mathbf{u}_{pert} - \Delta L_{pre,i})
$$

The last term is the perturbed member's own contribution to the supports and
must not be dropped; it carries the *mechanical* elongation because the
member's thermal force scales with its perturbed stiffness.
$\mathbf{F}_{ext}[\text{fixed}]$ is exactly zero for the common case of
mechanical loads on free nodes and no imposed strain — and exactly *not* zero
for a heated restrained structure, which is why `ReactionInfluence` carries it
as `f_ext_fixed`.

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
| Thermal demand in the fire chain | ✓ closed forms (restrained bar, free expansion) | ✓ (engine ≡ assembler path) | ✓ ($\alpha = 0$ reduction is exact) | ✓ (rank-1 ≡ re-solve, heated) | ✓ |
| Uniform-force scan (`UniformForceScan`) | — | ✓ (≡ per-point `member_axial_forces`) | ✓ (one factorisation per scan, counted) | — | ✓ |
| Uniform-$T$ ranking invariance | ✓ (proved, with the §5.4 precondition) | — | ✓ + antifake providers | — | ✓ |
| OpenSeesPy cross-check | — | ✓ (optional extra) | — | — | ✓ |
| Geometric stiffness $\mathbf{K}_G$ (§9) | ✓ (single-member golden) | ✓ (dense QZ eigensolve) | ✓ ($b \perp g$, energy identity, rigid motion) | — | ✓ (prestressed base state) |
| Bifurcation load factor (§9) | ✓ (toggle closed form, 4 geometries, rel $10^{-10}$) | ✓ (QZ on 8 campaign topologies) | ✓ (load scaling, tension-only ∞, zero-force irrelevance) | ✓ (mode residual $<10^{-10}$) | ✓ (imposed compression erodes λ) |
| ISO 834 curve (§10) | ✓ (6 published table points ±1 °C) | — | ✓ (monotone, $t=0$ anchor) | — | — |
| Lumped-capacitance heating (§10) | ✓ (constant-property exponential limit, rel $10^{-10}$) | ✓ (`solve_ivp` RK45, 0.01 °C) | ✓ (monotone, gas bound, $A_m/V$ & $k_{sh}$ orderings, step-halving) | — | ✓ (energy balance rel $2·10^{-4}$) |
| Secant thermal strain $\bar\alpha$ (§10.3) | ✓ (fixture polynomials) | — | ✓ ($\bar\alpha·Δθ ≡ ε_{th}$ dense grid) | — | ✓ |
| Reliability $\chi$-margin (§11.1) | — | ✓ (≡ limitstates capacity) | ✓ (EULER_ONLY bit-for-bit legacy) | — | ✓ (fire curve above ambient) |
| Empirical $p_f$ + Clopper–Pearson (§11.2) | ✓ (exact binomial bounds) | — | ✓ (count identity, $k=0$ band) | — | — |
| Iman–Conover correlation (§11.3) | — | — | ✓ (exact permutation/stratification, realised $\rho_S$ within noise) | — | — |
| `failure_mode` labels (§11.4) | ✓ (unloaded tie collapses at $k_E=0$) | — | ✓ (legacy facades bit-for-bit) | — | ✓ |
| Dimensional similarity | — | — | ✓ ($s$, $s^2$ scaling oracle, thermal variant) | — | ✓ |

Gaps that are honestly open: no experimental benchmark (e.g. the Cardington
fire tests), no FORM/SORM for small failure probabilities, no *nonlinear*
post-buckling / snap-through or material-nonlinear solve (the linearised
bifurcation check of §9 is now covered; the nonlinear continuation is not),
no transient heat conduction through the section (the lumped-capacitance
model of §10 assumes a uniform member temperature, as EN 1993-1-2 §4.2.2.2
itself does for unprotected members), and the OpenSeesPy bridge is an
optional extra so public CI does not run it.

---

## 9. Geometric Stiffness and Linearised Bifurcation (`stability.py`)

### 9.1 Why a first-order solver needs a stability companion

The static chain is first-order: forces come from $\mathbf{K}_E\mathbf{u} =
\mathbf{F}$ assembled on the *undeformed* geometry. Axial force nonetheless
changes a pin-jointed assembly's transverse stiffness — compression softens
it, tension stiffens it. That second-order effect lives in the **geometric
stiffness**

$$
\mathbf{K}_G = \sum_e \frac{N_e}{L_e}\,\mathbf{g}_e \mathbf{g}_e^T,
\qquad
\mathbf{g}_e\cdot\mathbf{u} = \text{transverse relative displacement of } e,
$$

with $\mathbf{g}_e = [s, -c, -s, c]$ on the member DOFs. Two structural facts
make this fit the existing formulation exactly:

* $\mathbf{g}_e \perp \mathbf{b}_e$ (the elastic elongation vector): the
  elastic dyad $k_e \mathbf{b}_e\mathbf{b}_e^T$ acts on the axial relative
  displacement, the geometric dyad on the transverse one. Verified to machine
  precision in `test_geometric_vectors_orthogonal_to_elastic_vectors`.
* $\mathbf{u}^T\mathbf{K}_G\mathbf{u} = \sum_e N_e L_e \varphi_e^2$ with
  $\varphi_e$ the chord rotation — twice the second-order work of the axial
  forces, the classical potential-energy statement
  (`test_geometric_stiffness_energy_identity`).

### 9.2 The bifurcation problem

Around the **prestressed base state** the tangent stiffness is
$\mathbf{K}_E + \mathbf{K}_G(\mathbf{N})$. Splitting the demand into imposed
(thermal/fabrication eigenstrain, held fixed) and mechanical (the load pattern
$\lambda$ amplifies),

$$
\big[\mathbf{K}_E + \mathbf{K}_G(\mathbf{N}_{imp}) + \lambda\,\mathbf{K}_G(\mathbf{N}_{mech})\big]\mathbf{u} = \mathbf{0},
$$

$\lambda_{cr}$ is the smallest positive root. With
$\mathbf{A} = \mathbf{K}_E + \mathbf{K}_G(\mathbf{N}_{imp}) = \mathbf{L}\mathbf{L}^T$
and $\mathbf{B} = -\mathbf{K}_G(\mathbf{N}_{mech})$ the problem becomes
$\mathbf{A}\mathbf{u} = \lambda\mathbf{B}\mathbf{u}$, solved through the
symmetric whitened form $\mathbf{C} = \mathbf{L}^{-1}\mathbf{B}\mathbf{L}^{-T}$,
$\nu = 1/\lambda$, so $\lambda_{cr} = 1/\nu_{\max}$ from one symmetric
eigensolve. $\mathbf{A}$ not positive definite ⇒ the imposed state *alone* has
already reached a critical point ⇒ `MechanismError`, never a silent factor.

### 9.3 Verification and scope

* **Closed form:** the shallow two-bar toggle has $P_{cr} = 2EAh^3/(b^2 L_0)$
  (symmetric mode) and $2EAb^2/(hL_0)$ (antisymmetric); the eigenproblem
  reproduces the smaller to $10^{-10}$ relative across four rise ratios.
* **Independent algorithm:** a dense LAPACK QZ generalised eigensolve
  (`scipy.linalg.eig(A, B)`) agrees with the whitened symmetric path on eight
  campaign topologies.
* **Prestress physics:** a redundant shallow fan with growing fabrication
  misfit shows $\lambda_{cr}$ falling monotonically (297 → 214 → 131 → 48 →
  6.4) and then the base state losing positive definiteness — precisely the
  restrained-thermal destabilisation a first-order DCR chain cannot see.

**Scope (deliberate):** system bifurcation of the pin-jointed assembly, *not*
member code checks (those remain the $\chi$ model of §4), *not* post-buckling,
snap-through or imperfection sensitivity. For shallow systems the true
collapse is a limit point the linearised factor approximates from the base
configuration ($O(\theta_0^2)$ apart for the toggle).

---

### 9.4 Eigen-path selection at scale (round-6 C1)

The bifurcation problem is posed on the free DOFs, so its cost is set by
$n = n_{\text{free}}$. The dense path is three $O(n^3)$ stages — Cholesky of
$A$, two triangular whitening solves, `eigh` — plus $O(n^2)$ memory for three
full matrices. The sparse path replaces all of them with one SuperLU
factorisation of $A$ plus $O(k)$ Lanczos iterations, sharing that
factorisation between the definiteness probe (`OPinv` at $\sigma = 0$) and
the eigensolve (`Minv`):

$$
B x = \nu A x, \qquad \nu_{\max} = 1/\lambda_{cr},
\qquad A = K_E + K_G(N_{\text{imposed}}), \quad B = -K_G(N_{\text{mech}}).
$$

Two details decide whether this is a scalability fix or a wrong answer:

* The request is the largest **algebraic** $\nu$ (`which="LA"`), not the
  smallest magnitude. Since $\nu = 1/\lambda_{cr}$, asking for `"SM"` would
  return the *highest* buckling load and label it critical.
* The definiteness probe shifts to $\sigma = 0$, not to $-\lVert A\rVert_F$.
  The far shift brackets the whole spectrum and looks more rigorous, but every
  transformed eigenvalue then clusters at $1/\lVert A\rVert_F$ and Lanczos
  cannot separate them: it failed to converge in 12811 iterations at
  $n = 1121$ where $\sigma = 0$ converges in under ten. The reduced scope of
  the $\sigma = 0$ probe — it detects the loss-of-definiteness *crossing*
  ($\lambda_{\min} \to 0$), which is the physically relevant failure, not a
  base state already deeply past it — is documented on
  `_sparse_smallest_eigenvalue`; use `eigen_solver="dense"` when such a state
  is suspected, since a Cholesky attempt gives an unconditional verdict.

Measured against the dense path on a Pratt truss (agreement $<2\times10^{-13}$
relative on $\lambda_{cr}$ and on mode subspace angles, with and without a
thermal prestress field):

| $n_{\text{free}}$ | 33 | 65 | 121 | 181 | 241 | 481 | 801 | 1281 | 2001 |
|---|---|---|---|---|---|---|---|---|---|
| sparse / dense time | 1.7× | 3.6× | 0.42× | 0.88× | 0.52× | 0.40× | 0.36× | 0.29× | 0.29× |

(Entries below 1 mean the sparse path is faster.) The crossover is near
$n \approx 100$, but `SPARSE_EIGEN_THRESHOLD` is set to **400** rather than
100: below that a solve costs well under a second, and the dense path buys two
things the sparse one cannot — an unconditional positive-definiteness verdict,
and the full spectrum rather than the leading Ritz pairs. `solver_path` on the
result reports which path actually ran, because `"sparse"` cannot be honoured
when ARPACK has no room ($k < n$) and that fallback should not be silent.

### 9.5 Imperfection sensitivity (round-6 B4)

A linearised $\lambda_{cr}$ is an upper bound, and how much of an upper bound
it is depends on the post-critical path. `imperfection_sensitivity` measures
that instead of asserting it: the perfect geometry's critical mode $\varphi$
is imposed on the *coordinates* at amplitude $\varepsilon$,

$$
x_i \leftarrow x_i + \varepsilon\, L_{\text{ref}}\, \varphi_i ,
$$

and the bifurcation analysis repeated on each imperfect geometry, giving the
first-order gradient $\mathrm{d}(\lambda/\lambda_0)/\mathrm{d}\varepsilon$
and a sensitivity verdict.

**Both signs of $\varepsilon$ are probed and the adverse one reported.** An
eigenvector's sign is arbitrary, and on an asymmetric post-critical path the
two signs are not equivalent: for a shallow toggle one sign deepens the arch
and $\lambda_{cr}$ grows roughly with the cube of the rise, while the other
flattens it and the reserve collapses. Reporting only the favourable sign
would be the single most dangerous output this function could produce, so
$\lambda_{cr} = \min(\lambda_+, \lambda_-)$ elementwise and both series are
retained for diagnosis. Each point is a fresh *linearised* analysis on the
imperfect geometry, not a limit-point search; full arc-length continuation
remains out of scope.

## 10. Fire Exposure and Member Heating (`thermal/fire_curve.py`)

### 10.1 The layering this closes

Before this module the library consumed a *prescribed* steel temperature and
produced structural response — "fire resistance for prescribed temperature
states", not fire analysis. The missing layer was exposure → member
temperature:

$$
\text{fire exposure } \theta_g(t)
\;\longrightarrow\; \theta_a(t) \text{ (member)}
\;\longrightarrow\; E(\theta_a), f_y(\theta_a)
\;\longrightarrow\; \text{structural response}.
$$

### 10.2 ISO 834 and lumped capacitance

The standard curve $\theta_g(t) = 345\log_{10}(8t+1) + \theta_0$ ($t$ in
minutes) reproduces the published table points (576/679/739/842/945/1049 °C at
5/10/15/30/60/120 min) to better than 1 °C. Member heating follows EN
1993-1-2 §4.2.2.2 for **unprotected** steel:

$$
\rho_a c_a(\theta_a)\,V\,\dot\theta_a = k_{sh} A_m\, h_{net},
\qquad
h_{net} = \alpha_c(\theta_g - \theta_a) + \varepsilon_{res}\sigma\big[(\theta_g{+}273)^4 - (\theta_a{+}273)^4\big],
$$

integrated with RK4 at ≤ 5 s steps (the code's own cap on $\Delta t$), with
$c_a(\theta_a)$ the standard's temperature-dependent specific heat — including
the endothermic spike near 735 °C — from the same Table 3.1 fixture the whole
material layer uses. Verification: constant-property convection limit matches
the closed-form exponential to $10^{-10}$; an independent `solve_ivp` RK45 at
tight tolerance matches to 0.01 °C; the surface heat input closes the enthalpy
balance $\rho_a\int c_a\,d\theta$ to $2·10^{-4}$.

**Scope:** uniform member temperature (no through-thickness gradient — the
lumped assumption the code itself endorses for unprotected members), gas
temperature given (no zone model), unprotected steel ($k_{sh}=1$ default;
protected members need the conduction solution and remain out of scope).

### 10.3 Secant thermal strain (the constant-$\alpha$ trap)

The standard defines thermal **elongation** $\Delta l/l = \alpha(\theta)$
(clause 3.4.1.1), a convex curve. The framework's prestress term
`alpha * delta_T * L` uses `Element.alpha` as a *constant* coefficient; with
the ambient value $1.2·10^{-5}$ this understates restrained thermal strain by
~21 % at 600 °C (secant slope $1.448·10^{-5}$). `thermal_strain(θ, θ₀)` and
`effective_alpha(θ, θ₀) = ε_{th}/(θ-θ₀)` expose the exact curve-based
quantities; setting `elem.alpha = effective_alpha(T)` makes the existing
prestress term reproduce the standard's elongation *exactly*
(`test_effective_alpha_reproduces_thermal_strain_exactly`). Element defaults
are deliberately unchanged: models without imposed strain stay bit-for-bit,
and the correction is an explicit, documented user choice.

---

### 10.4 Parametric fire — EN 1991-1-2 Annex A (round-6 B2)

ISO 834 is a prescriptive curve that rises forever and never cools. A
performance-based design needs a fire derived from the compartment, with a
real peak and a decay. With $t^* = t\,\Gamma$ in hours:

$$
\theta_g = 20 + 1325\left(1 - 0.324 e^{-0.2 t^*} - 0.204 e^{-1.7 t^*}
- 0.472 e^{-19 t^*}\right), \qquad
\Gamma = \frac{(O/b)^2}{(0.04/1160)^2}
$$

$$
t_{\max} = \max\!\left(\frac{0.2\times10^{-3} q_{t,d}}{O},\; t_{\lim}\right),
\qquad t^*_{\max} = t_{\max}\,\Gamma
$$

and, for $t^* > t^*_{\max}$, a cooling branch whose rate is selected by
$t^*_{\max}$: $625$ K/h below $0.5$ h, $250(3 - t^*_{\max})$ between $0.5$
and $2$ h, $250$ K/h above, with $\theta_g$ floored at $\theta_0$.

Two details are easy to get wrong and invisible in the curve's shape, which is
why the implementation is pinned against the Access Steel worked example
SX042a-EN-EU rather than against a recollection of the clause:

* $\Gamma$ is formed from the **opening factor** $O = A_v\sqrt{h_{eq}}/A_t$,
  not from $q_{t,d}$. The example gives $\Gamma = 5.791$; a $q_{t,d}$-based
  reading gives $6.04$ for the same compartment.
* $t_{\max}$ takes the **maximum** of the ventilation-controlled duration and
  $t_{\lim}$ (Table A.2: slow 25, medium 20, fast 15 min), so a lightly
  loaded or well ventilated compartment is fuel-controlled and bounded below.

Reproduced and pinned: $\Gamma = 5.791$, $t_{\max} = 0.355$ h,
$t^*_{\max} = 2.056$ h, $\theta_{\max} = 1052\,^\circ\mathrm{C}$, cooling
rate $250$ K/h, and the heating branch against a hand evaluation of A.1(1) to
$10^{-12}$. Limits of validity are enforced ($0.02 \le O \le 0.2$ m$^{0.5}$,
$100 \le b \le 2200$ J m$^{-2}$ s$^{-0.5}$ K$^{-1}$) and $q_{t,d}$ outside
$[50, 1000]$ MJ/m$^2$ warns as an extrapolation of the standard's fitting
data.

*Not implemented, deliberately:* the standard's **note** to A.1(1) defines a
different $\Gamma = (b/1160)^2/(q_{t,d}/420)^2$ for low-fire-load offices with
$b > 1200$. It is a National-Annex-dependent special case, and shipping an
unvalidated second variant of a fire curve is worse than not shipping it.

### 10.5 Protected members — EN 1993-1-2 §4.2.5.2 (round-6 B1)

For an insulated member the protection layer's own heat capacity enters,
because it must be heated before the steel behind it warms:

$$
\Delta\theta_a = k_{sh}\frac{\lambda_p}{d_p}\frac{A_p/V}{\rho_a c_a}
\,\frac{\theta_g - \theta_a}{1 + \mu/3}\,\Delta t
\;-\;\left(e^{\mu/10} - 1\right)\Delta\theta_g,
\qquad
\mu = \frac{c_p \rho_p}{c_a \rho_a}\, d_p \,\frac{A_p}{V}
$$

with $\Delta\theta_a \ge 0$ imposed whenever $\Delta\theta_g > 0$, and
$\Delta t \le 30$ s.

**The integrator is explicit Euler, not RK4** — deliberately, and against the
rest of this module. §4.2.2.2 states a continuous ODE and is answered with
RK4; §4.2.5.2 states a *recursion* with a non-negativity clip and a step
ceiling. Reproducing the code's answer means reproducing its recursion, so a
test measures the convergence order and asserts $\approx 1$: "improving" this
to RK4 fails the suite, because a fire-resistance duration quoted against a
different integrator than the code's is not a code-compliant duration.

The clip is load-bearing on the very first step. ISO 834 jumps from 20 °C to
~261 °C in 30 s, so the $-(e^{\mu/10}-1)\Delta\theta_g$ time-delay term
outweighs a heating term that starts from $\theta_g - \theta_a = 0$ and the
raw increment is *negative*; without the clip a protected member would predict
a temperature below ambient while the fire is climbing. During a decaying
(parametric) fire the clip does not apply and the member cools, which a test
pins so the model cannot become a ratchet.

$k_{sh}$ defaults to $1.0$ (contour protection), the conservative choice since
it multiplies the heating term; `box_protection_shadow_factor` computes the
$0.9\,(A_p/V)_{box}/(A_p/V)$ reduction of §4.2.5.2(2) and rejects reversed
arguments, which would otherwise give $k_{sh} > 1$ — a less conservative
temperature from a call that looks valid.

The shipped material catalogue is a **secondary source**: representative
literature values, not code-mandated numbers and not product data. It carries
no temperature dependence of $\lambda_p$ or $c_p$, so gypsum's endothermic
dehydration plateau near 100–200 °C is absent and a gypsum-protected member is
predicted to heat *faster* than it really does — the safe direction, but an
approximation. No intumescent coatings (whose effective thickness is a
function of temperature and time), no cavity or multi-layer build-up.

### 10.6 When the lumped assumption stops holding (round-6 B7)

Both heating models assume a uniform cross-section temperature. That is valid
while the Biot number is small:

$$
Bi = \frac{h_{\text{eff}}\,(V/A_m)}{\lambda_a} \ll 0.1
$$

`lumped_capacity_biot` derives $h_{\text{eff}}$ from the module's own
`h_net` linearised about a representative exposure (800 °C gas against the
member's initial temperature), so it carries the radiative term that dominates
a real compartment fire rather than assuming a constant film coefficient.
$\lambda_a$ is evaluated at that reference temperature, not at ambient: steel
conductivity falls with temperature and the lower value gives the larger,
conservative Biot number.

`steel_temperature` warns below $A_m/V = 50\ \mathrm{m}^{-1}$, the reference
threshold for a "thick" section. The message quotes both numbers and their
relationship, because the classical $Bi < 0.1$ criterion is reached at
$A_m/V \approx 34\ \mathrm{m}^{-1}$ for this exposure — i.e. the code
threshold is the *conservative* of the two and fires first. A warning that
quoted a Biot number passing its own stated criterion (what a bare
50-threshold warning does at $A_m/V = 45$) would be self-contradicting.

## 11. Reliability-layer conventions (`reliability.py`, `uncertainty/sampling.py`)

### 11.1 One capacity model everywhere

The buckling safety margin is $\chi A f_y/\gamma_M - |N|$ with $\chi$ from the
*same* `sections.buckling_reduction_factor` the DCR chain uses (fire
imperfection factor $0.65\alpha$ above ambient, EN 1993-1-1 curve at 20 °C),
cross-pinned against `limitstates._member_limit_state` to $10^{-12}$. The
pre-2.8 bare-Euler margin was optimistic by up to ~6× at $\bar\lambda \approx
0.5$; `BucklingModel.EULER_ONLY` reproduces legacy numbers bit-for-bit.
`MemberResponse.E` and `.yield_stress` are taken **as given** (the analysis
callback decides on temperature reduction); `.temperature` only selects the
code regime for $\chi$.

### 11.2 Empirical failure probability

`pf_approx = Φ(−β̂)` assumes a normal margin; margins built from skewed
inputs (lognormal $f_y$, Gumbel load) are skewed exactly where $p_f$ lives.
`pf_empirical = mean(margin < 0)` is assumption-free but resolution-limited;
its Clopper–Pearson 95 % interval (`pf_empirical_ci`) is the honest output —
at zero observed failures the upper bound is $\approx 3/n$, the study's
resolution, not a claim of safety.

### 11.3 Rank correlation without destroying the design

`sample_spec_matrix` correlates its Latin hypercube through **Iman–Conover**
rank reordering: every output column is an exact permutation of the input
column, so stratification (the variance reduction LHS exists for) survives,
while the realised Spearman correlation reproduces the target within sampling
noise. The Gaussian-copula linear mix remains available for non-LHS inputs;
its docstring now states plainly that mixing destroys stratification. Both
paths share one target semantics (Spearman, Kruskal-mapped to normal-space
Pearson before factorisation) and one input validation (symmetric, unit
diagonal, positive definite).

### 11.4 Critical temperatures carry their failure mode

`member_critical_temperature_detailed` /
`system_critical_temperature_detailed` return
`(theta, failure_mode ∈ {material, stiffness_collapse, none})`. A lightly
loaded member whose scan ends at the $k_E = 0$ table endpoint is reported as
`stiffness_collapse`, never as a material critical temperature — the two have
different engineering meanings. The legacy scalar facades delegate and remain
bit-for-bit identical. Under the real Eurocode law $k_y$ reaches zero before
$k_E$, so any *loaded* member fails in `material` mode first; `stiffness_
collapse` is reached by essentially unloaded members or grids that include
1200 °C.

---

## 12. Physics Boundary — what this solver is and is not

The round-5 audit's strongest process demand: state the validity envelope
formally, not implicitly.

| Phenomenon | Status | Where |
|---|---|---|
| Linear truss statics (small displacement) | exact within numerical tolerance | §1, full verification matrix |
| Thermal / fabrication eigenstrain | supported, closed-form verified | §1, restrained & free limits |
| Temperature-dependent $E(\theta), f_y(\theta)$ | supported (Table 3.1 single source) | §4, material goldens |
| Secant thermal strain $\bar\alpha(\theta_0,\theta)$ | supported (opt-in via `effective_alpha`) | §10.3 |
| Fire exposure → member temperature | supported (ISO 834 + lumped capacitance, unprotected, uniform section temperature) | §10 |
| Fire resistance at prescribed temperature | supported (EN 1993-1-2 §4.2.3.1 $\chi$ model) | §4 |
| Member buckling capacity | supported (Euler/$\chi$, flexural, fire & ambient curves) | §4 |
| System bifurcation under prestress | supported (linearised) | §9 |
| Reliability (crude MC, LHS, rank correlation) | supported; rare-event methods not | §11 |
| Thermal transient conduction through the section | **not supported** (uniform temperature only); validity now *measured* via the Biot number and warned below $A_m/V = 50\ \mathrm{m}^{-1}$ | §10.6 |
| Protected / insulated members | supported (EN 1993-1-2 §4.2.5.2, explicit-Euler recursion, single homogeneous layer) | §10.5 |
| Parametric (natural) fire curves | supported (EN 1991-1-2 Annex A main clause; the A.1(1)-note office variant is not) | §10.4 |
| Geometric nonlinearity (P-Δ, large displacement) | **not supported**; validity guarded by `LargeDisplacementWarning` and diagnosable via §9 | §9.3 scope |
| Imperfection sensitivity | supported as a *linearised sweep* over imposed geometric imperfections, both signs, adverse reported | §9.5 |
| Post-buckling / snap-through / arc-length continuation | **not supported** | §9.3, §9.5 |
| Large-model bifurcation ($n_{\text{free}} \ge 400$) | supported via sparse Lanczos; agreement with the dense path $<2\times10^{-13}$ | §9.4 |
| Material nonlinearity (plasticity, redistribution) | **not supported** (capacity checks are code-model, not incremental analysis) | §4 |
| Creep & transient thermal strain | **not supported** (deferred, see below) | — |
| Torsional / torsional-flexural buckling | **not supported** (needs $I_z, I_t, I_w$ absent from the section model) | — |
| 3D, frames, semi-rigid joints, distributed loads | **not supported** (2D pin-jointed, nodal loads) | §1 |
| FORM/SORM, PCE, Sobol indices, subset simulation | **not supported** (deferred) | §11.2 |
| Accuracy of the first-order tangent $K_E + K_G$ | measured, not assumed: `verify_linearization_convergence` reports the gap to the exact tangent closing at fitted order 1.000 | `tangent_verification` |
| Validation against physical fire or structural test data | **not supported** — verification is against closed forms, independent algorithms and code worked examples; no experimental corpus ships, so agreement with the real world is inferred from agreement with the standards, not measured | §12 |

**Deferred with rationale (round-5):** creep/transient-strain models and
torsional-flexural buckling need material/section data and validation sources
beyond the current fixture; PCE/Sobol/subset simulation constitute a UQ
subsystem with its own validation burden; nonlinear continuation (Newton–
Raphson arc-length) supersedes — not extends — the linearised §9 check and
belongs to a dedicated release. Each deferral is recorded in CHANGELOG 2.8.0
so the boundary is versioned, not vibes.

**Round-6 additions to the boundary:** protected-member heating (§10.5) and
parametric fire curves (§10.4) move two rows from *not supported* to
*supported*, each with its own stated limits. Imperfection sensitivity (§9.5)
is supported as a linearised sweep — it quantifies how far $\lambda_{cr}$ can
be trusted without becoming a post-buckling analysis, which is still out.
Exact-tangent verification (`tangent_verification.py`) does not extend the
boundary but *measures* it: `verify_linearization_convergence` reports that the
relative Frobenius gap between the library's $K_E + K_G$ and the exact tangent
closes at fitted order **1.000** over a 16× load range, which is the precise
sense in which §9 is a linearised theory.
