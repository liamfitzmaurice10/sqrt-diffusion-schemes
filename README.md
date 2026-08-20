# Numerical Schemes for Square-Root Diffusions

Introduction

## Background

The Cox-Ingersoll-Ross process

$$dS(t) = \kappa(\lambda - S(t)) dt + \sigma\sqrt{S(t)}dB(t), \qquad S(0) = S_0 > 0$$

has a diffusion coefficient that is not globally Lipschitz, so the standard convergence theory for Euler-Maruyama does not apply.
While the Feller condition, $2\kappa\lambda \geq \sigma^2$, holding means the true process will be positive almost surely,
naive discretisations can produce negative values under the square root regardless of the condition being met.
Feller not being met simply increases the rate at which the scheme may go negative.  

This is not merely academic. CIR is both a short-rate model and the variance process of the Heston model.
Heston parameters calibrated to equity index options routinely violate Feller by a wide margin, the variance parameters used in the pricing section below have Feller ratios of 0.75 and 0.33. 
What matters in practice is how each scheme behaves near zero at realistic stepsizes, not just its convergence order in the limit.

## Schemes compared

These four schemes were selected for comparison for three distinct reasons: 
the reflected scheme as a natural way of enforcing positivity by construction; 
the absolute and truncated schemes because they are known to converge strongly to the true CIR solution as $h \to 0$ 
(Dereich, Neuenkirch & Szpruch, 2012, and related work); 
and the Lamperti scheme because it is drift-implicit and attains strong order 1 under the Feller condition, making its behaviour once that condition fails particularly interesting.

Write $h$ for the stepsize and $\Delta B_{n+1}$ for the Brownian increment.

The Lamperti transform $Y = \sqrt{S}$ gives
$dY = (\alpha/Y + \beta Y)dt + \gamma dB$ with
$\alpha = (4\kappa\lambda - \sigma^2)/8$,  $\beta = -\kappa/2$, $\gamma = \sigma/2$.

| Scheme | Update | Non-negativity |
|---|---|---|
| Explicit EM (absolute) | $S_{n+1} = S_n + h\kappa(\lambda - S_n) + \sigma\sqrt{\lvert S_n\rvert} \Delta B_{n+1}$ | No, the state may go negative, but the diffusion then acts on $\lvert S_n\rvert$ |
| Explicit EM (truncated) | $S_{n+1} = S_n + h\kappa(\lambda - S_n) + \sigma\sqrt{\max(S_n,0)} \Delta B_{n+1}$ | No, the state may go negative, but the diffusion then switches off and only the drift $\kappa\lambda h$ acts |
| Explicit EM (reflected) | $S_{n+1} = \lvert S_n + h\kappa(\lambda - S_n) + \sigma\sqrt{S_n} \Delta B_{n+1}\rvert$ | Yes, by construction |
| Lamperti implicit | $Y_{n+1} = \dfrac{u}{2(1-\beta h)} + \sqrt{\dfrac{u^2}{4(1-\beta h)^2} + \dfrac{\alpha h}{1-\beta h}}$, $u = Y_n + \gamma\Delta B_{n+1}$ then $S_{n+1} = Y_{n+1}^2$ | Yes when $\alpha \geq 0$. Otherwise undefined as the radicand can go negative causing the scheme to return NaN |

The first three coincide exactly whenever a path stays positive, so any difference between them is a direct measurement of how often the discretised process crosses zero. 
Lamperti remains well defined whenever $\alpha \ge 0$ $(4\kappa \lambda \geq \sigma^2) $, a strictly weaker condition than Feller. 
Thus, in the range $2\kappa\lambda < \sigma^2 \le 4\kappa\lambda $, the Feller condition is not met and the true process can reach zero, yet the Lamperti scheme will not produce NaN. 

## Method

- The CIR allows exact transition sampling through the scaled noncentral $\chi^2$ distribution and sod closed forms are used for the weak-error references: $E[S(T)]$ and $E[(S(T)-K)^+]$.
  Both are exact, so there is no Monte Carlo noise on the reference side.

## Results

### Weak error
![Weak error](figures/02_weak_error.png)
*The dashed "slope 1" is the rate a standard Euler–Maruyama scheme would attain under globally Lipschitz coefficients. 
The coefficients here are not globally Lipschitz, so the line is included only as a visual benchmark.*

At the baseline $\sigma = 0.1$, $q(x) = x$ no scheme has a resolvable weak error.
Every measurement sits within 2–8 standard errors of zero and is flat in $h$ (fitted slopes 0.00 to 0.40). 
For the absolute and truncated schemes this is exact, not just small.
$S_0 = \lambda$ is the fixed point of the mean recursion, so $E[S_n] =\lambda$ at every step-size, matching $E[S(t)] = \lambda$ exactly.
Reflected coincides with them here, since paths don't approach zero at this $\sigma$.
Lamperti's error is not exactly zero as it carries its own small bias from the transform but for $\sigma = 0.1$, it is too small to distinguish from the Monte Carlo noise.


For $\sigma = 0.1$, $q(x) = (x-0.2)^+$, the weak error is real and resolvable at the coarse end but shrinks sharply as $h$ decreases.
By the middle of the tested range it appears to have fallen to the size of the Monte Carlo noise floor. 

At $\sigma = 1$ the Feller condition fails and the behaviour changes. 
For $q(x) = x$, the absolute and truncated schemes' error remains exactly zero for the same reasons as it did before.
Reflection now injects mass at the boundary, showing a mean bias of $4.3\times10^{-2}$, 65 standard errors, decaying with fitted slope 0.51.
For $q(x) = (x-0.2)^+$, all three explicit schemes have non-zero weak error, 
with fitted weak orders 0.68 (absolute), 0.62 (truncated), 0.57 (reflected).


The Lamperti curves at $\sigma = 1.0$ are flat at around $1.5\times10^{-1}$ but should not be read as weak errors.
55% of paths fail, and the average is taken over the survivors, the paths that avoided the region close to zero where the scheme breaks.

### Strong convergence under the Feller condition


### Effect on Heston option prices


## Limitations

## References

