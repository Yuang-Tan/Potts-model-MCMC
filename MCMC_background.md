# Why Monte Carlo works for interacting spins

## 1. The computational problem is the measure

An equilibrium observable is an average over microscopic configurations, weighted by their physical probabilities. For a classical system in contact with a heat bath, the canonical distribution is

$$
\pi(x)=Z^{-1}e^{-\beta E(x)},\qquad
\langle A\rangle=\frac{\sum_x A(x)e^{-\beta E(x)}}{\sum_x e^{-\beta E(x)}},\qquad
\beta=(k_BT)^{-1}.
$$

The difficulty is the number of terms, not the appearance of the formula. An Ising lattice with N spins has 2 to the power N configurations; a three-state Potts lattice has 3 to the power N. Increasing the lattice width therefore makes exhaustive enumeration exponentially more expensive. Computing the energy of one configuration can be easy while computing its contribution relative to every other configuration remains impractical.

Enumeration also ignores the uneven distribution of physical relevance. Low-energy states have large individual Boltzmann weights, but numerous higher-energy configurations can collectively dominate through entropy. Equilibrium consequently reflects a competition between energetic preference and multiplicity. Sampling only a few favourable configurations would confuse energy minimisation with thermal averaging; sampling configurations uniformly would usually spend most effort far from the important part of the equilibrium distribution.

The numerical objective is to estimate the average without explicitly constructing the complete probability distribution. Monte Carlo replaces an exhaustive sum with a statistical estimate. Its usefulness depends on how samples are distributed and how much independent information they contain, rather than simply how many configurations the program visits.

## 2. Random sampling gives an estimator, not automatic efficiency

For an integral over a region of volume V, independent uniform samples give

$$
I=\int_\Omega f(x)\,dx,\qquad
\widehat I=\frac{V}{n}\sum_{i=1}^{n}f(X_i),\qquad
\operatorname{Var}(\widehat I)=\frac{V^2}{n}\operatorname{Var}[f(X)].
$$

The estimator is unbiased because the uniform probability density is the reciprocal of the volume. With finite variance, its standard error decreases as the inverse square root of the sample count. Halving the error therefore requires roughly four times as many independent samples. This scaling does not explicitly depend on dimension, but the variance and the cost of drawing useful samples can depend strongly on it. Monte Carlo does not remove the difficulty of a poorly matched sampling distribution.

Uniformity must also refer to the correct measure. For example, choosing radius and angle independently and uniformly does not give uniform points in a disc: outer annuli occupy more area. The radius must instead be proportional to the square root of a uniform random number. The same principle applies to configuration space: an apparently random construction need not sample the probability measure required by the observable.

For Boltzmann averages, uniformly generated configurations can be reweighted by their Boltzmann factors. The numerator and denominator then estimate separate sums. Their ratio is generally biased at finite sample count, even when each sum is estimated without bias; under suitable conditions it is consistent as the sample count grows. More seriously, a few rare configurations may dominate the weights. Repeatedly generating typical uniform configurations then adds little information about the desired thermal average.

This problem is especially severe when the equilibrium and uniform ensembles concentrate on different energy ranges. For a large lattice, the configurations typically produced by uniform labels can be exponentially unlikely under the target measure. Reweighting cannot recover a region that was never visited. Weight concentration is therefore a diagnostic of poor overlap, not merely an inconvenience caused by numerical underflow. Computing logarithmic weights can stabilise arithmetic, but it cannot create the missing information.

## 3. Importance sampling changes where effort is spent

Choose a normalised density p that is positive wherever the integrand contributes. Rewriting the integral gives

$$
I=\mathbb E_p\!\left[\frac{f(X)}{p(X)}\right],\qquad
\operatorname{Var}(\widehat I_p)=\frac1n\left[\int_\Omega\frac{f(x)^2}{p(x)}\,dx-I^2\right].
$$

The inverse probability factor compensates for sampling some regions more frequently. The variance expression makes the design problem explicit: assigning very small probability where the integrand is large can make the estimate unstable. For independent sampling, a density proportional to the absolute integrand minimises this variance, but usually requires information about the unknown integral. Practical importance sampling seeks a useful approximation rather than a universally optimal distribution.

For equilibrium physics, the natural target is the Boltzmann distribution itself. If configurations are sampled from it, the estimator is simply the average of the observable; no additional Boltzmann reweighting is required. This is not necessarily the lowest-variance choice for every observable, but it samples the physical ensemble and supports the estimation of many observables from the same configurations.

The remaining obstacle is that the partition function is unknown. Directly drawing independent configurations from the target can be as difficult as evaluating the original sum. Markov chain Monte Carlo avoids this demand by constructing local transitions whose stationary distribution is the desired ensemble. It replaces the problem of drawing independent global configurations with the problem of exploring configuration space sufficiently well.

## 4. Local transitions can preserve a global distribution

A Markov chain specifies the probability of its next configuration using its current configuration. Let g propose a candidate and let a accept it. For distinct configurations, the transition probability is their product. Detailed balance requires probability flow between every pair to be symmetric at stationarity:

$$
\pi(x)g(y\mid x)a(y\mid x)=\pi(y)g(x\mid y)a(x\mid y).
$$

Summing this relation over the departure states, using normalised transition probabilities, shows that the target distribution is stationary. Stationarity and convergence are different statements. A chain initialised from the target retains that distribution; a chain initialised arbitrarily must first approach it. For a finite state space, irreducibility and aperiodicity ensure convergence to a unique stationary distribution, but do not specify a practically acceptable mixing time.

The Markov property does not imply independent samples: dependence on the current state propagates information about earlier states through the sequence. Detailed balance is a convenient sufficient condition for stationarity, rather than a necessary property of every valid sampler. At zero temperature, accepting only non-increasing energies can destroy accessibility between configurations. Claims about eventual exploration therefore require attention to temperature, proposal support and the resulting transition graph.

The Metropolis–Hastings acceptance rule enforces detailed balance:

$$
a(y\mid x)=\min\!\left[1,\frac{\pi(y)g(x\mid y)}{\pi(x)g(y\mid x)}\right]
=\min[1,e^{-\beta\Delta E}]
\quad\text{for symmetric proposals},
$$

where the energy change is the proposed energy minus the current energy. The partition function cancels in the probability ratio. A local energy change can therefore be sufficient even though the target distribution depends on the energy of an entire many-body configuration. Energy-lowering proposals are accepted; energy-increasing proposals remain possible and are necessary for thermal fluctuations.

Rejection is part of the chain, not a failed observation to discard. All rejection probabilities contribute to the probability of staying in the current state. Recording only accepted configurations would change their statistical weights by ignoring how long the chain remains in them. Similarly, an asymmetric proposal requires the proposal-probability ratio: retaining the simple Metropolis factor without that correction generally changes the stationary distribution.

## 5. What the spin update actually estimates

In the Ising example, proposing to flip a uniformly selected spin is symmetric. For the three-state Potts model, select a site uniformly and choose either alternative label with equal probability. The reverse proposal has the same probability. On a square lattice, the energy change depends only on four neighbours and the external field:

$$
\Delta E=-\frac{3J}{2}(n_{\rm new}-n_{\rm old})-h(s_{{\rm new},x}-s_{{\rm old},x}).
$$

The neighbour counts refer to agreement with the proposed and current labels. The constant energy offset between the vector and Kronecker-delta formulations cancels. A sweep comprises N attempted updates, making computational effort comparable across lattice sizes. Random-site selection does not guarantee that every site is visited during each sweep.

The measurement definition matters as much as the update rule. In zero field, symmetry can make the ensemble-average magnetisation vector vanish even when individual configurations are strongly ordered. Averaging the magnitude of each configuration's magnetisation avoids cancellation between orientations. Yet the magnitude remains positive in a finite disordered lattice because random contributions rarely cancel exactly. It must therefore be interpreted together with temperature and system size.

A field adds another distinction: a large magnetisation magnitude indicates alignment among spins, but not necessarily alignment with the field. Its directional component can reveal a chain trapped in an unfavoured orientation. The physical meaning of a curve follows from the observable being measured and the ensemble actually sampled, rather than from its shape alone.

The chain also introduces an algorithmic time coordinate. A sweep measures computational work, but the chosen proposal and acceptance rules do not automatically reproduce a material's microscopic dynamics. Equilibrium averages can be physically meaningful even when relaxation times are algorithm-dependent. A dip observed during numerical heating should therefore first be examined as a property of the sampling protocol before being interpreted as a physical response rate.

## 6. The limiting resource is independent exploration

Local updates generate correlated configurations. For a stationary observable with finite variance and a summable autocorrelation function, the long-run uncertainty can be written approximately as

$$
\operatorname{Var}(\overline A)\simeq\frac{2\tau_{\rm int}}{n}\operatorname{Var}(A),\qquad
\tau_{\rm int}=\frac12+\sum_{t=1}^{\infty}\rho_A(t),\qquad
n_{\rm eff}\simeq\frac{n}{2\tau_{\rm int}}.
$$

Thus ten thousand measurements need not contain ten thousand independent observations. The correlation time depends on the observable as well as the algorithm and physical parameters. Near a critical point, rearrangements occur over growing length scales, while a local algorithm changes only one spin at a time. This mismatch produces critical slowing down. A faster implementation increases the number of attempted updates per second but does not remove these correlations.

Metastability creates a related problem. A chain may fluctuate within one region of configuration space while rarely crossing the barrier to another. Its trace can look stable and its estimated error small even though important states remain unsampled. An ordered initial state can retain its orientation; a random initial state can reduce that particular bias, but it is not automatically an equilibrium state at finite temperature. Heating also carries previous configurations forward, so an apparent transition can depend on the entire temperature schedule.

The relevant barrier can involve both energy and entropy. Many configurations may belong to the same macroscopic basin, and a narrow set of connecting configurations can obstruct exploration even without an obvious energetic obstacle in a chosen projection. Examining only the total energy may therefore miss slow motion in an order parameter. Diagnostics should track observables associated with the competing configurations.

Useful checks compare independent runs, different initial conditions, longer discarded periods and longer measurement windows. Exact enumeration of small systems tests implementation; agreement there does not establish equilibration of larger systems. Standard deviation describes the spread of observations, whereas standard error describes uncertainty in their mean. Using independent run-level means accommodates within-run correlation, but their uncertainty still cannot expose a bias shared by every run.

Algorithmic improvements must preserve the target distribution. Cluster moves can change correlated regions together, while replica exchange can help configurations traverse barriers through exchanges between temperatures. Such exchanges need an acceptance rule derived from the joint distribution. Annealing instead changes the target as temperature varies, so it should not be assumed to produce equilibrium samples at each step. In every case, the relevant performance measure is reliable information about the physical ensemble per unit computation.

## Source

Richard G. Morris, *Honours Statistical Physics 4142: Assignment 1*, 10 March 2026, sections “Motivation” through “Drawbacks of MCMC”. This companion develops the statistical reasoning behind the supplied notes; the separate notebook contains the simulation results.
