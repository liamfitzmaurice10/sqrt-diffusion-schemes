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



## Results

### Weak error

### Strong convergence under the Feller condition


### Effect on Heston option prices


## Limitations

## References

