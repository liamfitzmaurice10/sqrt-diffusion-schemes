# Numerical Schemes for Square-Root Diffusions

Introduction

## Background

The Cox-Ingersoll-Ross process

$$dS(t) = \kappa(\lambda - S(t))dt + \sigma\sqrt{S(t)}dB(t), \qquad S(0) = S_0 > 0$$

has a diffusion coefficient that is not globally Lipschitz, so the standard convergence theory for Euler-Maruyama does not apply.
While the Feller condition, $2\kappa\lambda \geq \sigma^2$, holding means the true process will be positive almost surely,
naive discretisations can produce negative values under the square root regardless of the condition being met.
Feller not being met simply increases the rate at which the scheme may go negative.  

This is not merely academic. CIR is both a short-rate model and the variance process of the Heston model.
Heston parameters calibrated to equity index options routinely violate Feller by a wide margin, the variance parameters used in the pricing section below have Feller ratios of 0.75 and 0.33. 
What matters in practice is how each scheme behaves near zero at realistic stepsizes, not just its convergence order in the limit.

## Schemes compared

The schemes
## Method



## Results

### Weak error

### Strong convergence under the Feller condition


### Effect on Heston option prices


## Limitations

## References

