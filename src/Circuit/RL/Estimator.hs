-- | Policy-gradient estimator axes for the REINFORCE == pathwise oracle.
--
-- The instance table (loom/instance-table.md §1) names two estimator axes:
--
-- * REINFORCE (score-function): @Prob (->) (Dual r)@ — the continuation
--   carries a gradient component that accumulates via the dual-number product
--   rule: @(v,g)·(v',g') = (v·v', v·g' + g·v')@.  The log-policy score
--   function injects gradient-weighted-by-reward exactly when the continuation
--   multiplies by a score morphism.
--
-- * Pathwise (reparametrization): the derivative is taken through the sample
--   @a = θ + σε@, so @∂R/∂θ = ∂R/∂a · 1@.
--
-- The oracle is: on a 1-step Gaussian policy @N(θ, σ²=0.25)@ with quadratic
-- reward @R(a) = -(a-1)²@, both estimators analytically reduce to
-- @∇J = -2(θ-1)@.  At @θ₀=3@ this is @-4@ — exact in Double.
module Circuit.RL.Estimator
  ( -- * Dual scalar for score-function gradient accumulation
    Dual (..),

    -- * Estimators
    reinforceGrad,
    pathwiseGrad,
    closedFormGrad,
  )
where

import Prelude

-- ---------------------------------------------------------------------------
-- Dual scalar — the score-function axis
-- ---------------------------------------------------------------------------

-- | Dual-number scalar: value plus accumulated gradient.
--
-- Multiplication follows the dual-number product rule: the gradient
-- component records the cross-derivative.  This naturally gives the
-- REINFORCE coupling: when a continuation multiplies a downstream reward
-- (value component) by a score function (gradient component), the result
-- records the reward-weighted score.
--
-- >>> Dual (2, 3) * Dual (5, 7)
-- Dual (10, 29)
--
-- Check: 2·5 = 10, 2·7 + 3·5 = 14+15 = 29 ✓
newtype Dual a = Dual {getDual :: (a, a)}
  deriving stock (Eq, Show)

instance (Num a) => Num (Dual a) where
  Dual (v1, g1) + Dual (v2, g2) = Dual (v1 + v2, g1 + g2)
  Dual (v1, g1) * Dual (v2, g2) =
    Dual (v1 * v2, v1 * g2 + g1 * v2)
  negate (Dual (v, g)) = Dual (negate v, negate g)
  abs = error "Dual: abs not defined"
  signum = error "Dual: signum not defined"
  fromInteger n = Dual (fromInteger n, 0)

instance (Fractional a) => Fractional (Dual a) where
  fromRational r = Dual (fromRational r, 0)
  recip (Dual (v, g)) =
    Dual (recip v, negate g / (v * v))
  {-# INLINE recip #-}

-- ---------------------------------------------------------------------------
-- Gaussian central moments (exact, polynomial)
-- ---------------------------------------------------------------------------

-- | The k-th central moment E[(a-μ)^k] for a ~ N(μ, σ²).  For symmetric
-- Gaussians, odd moments vanish.  Even moments: E[z^(2m)] = σ^(2m)·(2m-1)!!.
--
-- >>> gaussMoment 0.5 0
-- 1.0
-- >>> gaussMoment 0.5 2
-- 0.25
-- >>> gaussMoment 0.5 4
-- 0.1875
gaussMoment :: Double -> Int -> Double
gaussMoment sigma k
  | odd k = 0
  | otherwise =
      let sigmaSq = sigma * sigma
          doubleFact n = product [n, n - 2 .. 2]
       in (sigmaSq ** fromIntegral (k `div` 2)) * fromIntegral (doubleFact (k - 1))

-- ---------------------------------------------------------------------------
-- Estimators
-- ---------------------------------------------------------------------------

-- | Closed-form gradient for Gaussian policy N(θ, σ²) with quadratic reward
-- @R(a) = -(a - a_target)²@.
--
-- @J(θ) = E[R(a)] = -(σ² + (θ - a_target)²)@, so @∇J = -2(θ - a_target)@.
--
-- >>> closedFormGrad 3.0 0.5 1.0
-- -4.0
closedFormGrad :: Double -> Double -> Double -> Double
closedFormGrad theta _sigma aTarget = -(2 * (theta - aTarget))

-- | REINFORCE (score-function) gradient estimator for Gaussian policy.
--
-- @∇J = E[R(a) · ∇_θ log π(a)]@ where @∇_θ log π(a) = (a-θ)/σ²@.
--
-- The analytical expectation uses Gaussian central moments.
-- Let @z = a-θ@, @δ = θ-a_target@:
--
-- @
--   R(a) = -(a - a_target)² = -(z+δ)²
--   R(a)(a-θ)/σ² = -(z+δ)²·z/σ² = -(z³ + 2δz² + δ²z)/σ²
--   E[R(a)(a-θ)/σ²] = -(1/σ²)(E[z³] + 2δ·E[z²] + δ²·E[z])
--                   = -(1/σ²)(0 + 2δσ² + 0) = -2δ
--                   = -2(θ-a_target)
-- @
--
-- >>> reinforceGrad 3.0 0.5 1.0
-- -4.0
reinforceGrad :: Double -> Double -> Double -> Double
reinforceGrad theta sigma aTarget =
  let sigmaSq = sigma * sigma
      delta = theta - aTarget
      -- E[-(a-a_target)²·(a-θ)/σ²] = -E[z³+2δz²+δ²z]/σ²
      -- where z = a-θ ~ N(0,σ²)
      cubicTerm = gaussMoment sigma 3 -- 0
      quadTerm = 2 * delta * gaussMoment sigma 2 -- 2δσ²
      linearTerm = delta * delta * gaussMoment sigma 1 -- 0
   in -((cubicTerm + quadTerm + linearTerm) / sigmaSq)

-- | Pathwise (reparametrization) gradient estimator for Gaussian policy.
--
-- @a = θ + σε@ with @ε ~ N(0,1)@.
-- @∂R/∂θ = ∂R/∂a · ∂a/∂θ = -2(a - a_target) · 1 = -2(θ+σε-a_target)@.
-- @E[∂R/∂θ] = -2(θ-a_target)@ since @E[ε] = 0@.
--
-- >>> pathwiseGrad 3.0 0.5 1.0
-- -4.0
pathwiseGrad :: Double -> Double -> Double -> Double
pathwiseGrad theta sigma aTarget =
  -- E[-2(θ+σε-a_target)] where ε ~ N(0,1), E[ε] = 0
  (-(2 * theta)) + (2 * aTarget) + (-(2 * sigma * gaussMoment 1 1))
