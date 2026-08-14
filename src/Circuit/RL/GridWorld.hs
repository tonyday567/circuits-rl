{-# LANGUAGE DerivingStrategies #-}

-- | A tiny gridworld for the circuits-rl frontier spike.
--
-- The demonstration pins the reward/policy design choice upfront:
--
-- * Reward is a function of state, applied via a local 'scoreBy' modality.
-- * Transitions are deterministic for exact hand-checking.
-- * Value iteration is shown both directly and as composition in 'Prob'.
module Circuit.RL.GridWorld
  ( -- * Gridworld
    State (..),
    Action (..),
    step,
    reward,

    -- * Direct value iteration
    bellmanPolicy,
    bellmanOpt,
    valueIter,
    optimalPolicy,

    -- * Prob-composition view
    scoreBy,
    transP,
    rewardP,
    bellmanP,
    backupP,

    -- * Discounted-return oracle
    discountedReturn,
    closedFormReturn,

    -- * System (Prob) view
    expectSystem,
    gridSystem,
    mdpSystem,
    mdpCheck,
    pomdpSystem,
    pomdpCheck,
    Observation (..),
    observe,
    bellmanSystem,
    valueIterSystem,

    -- * Tropical / shortest-path row
    Tropical (..),
    shortestPath,
  )
where

import Circuit.Category (id, (.))
import Circuit.Poly (Mono, Poly (..), System, monoDir, monoIn, runSystem, system)
import Circuit.Prob (Prob (..), embed, score)
import Data.List (foldl', maximumBy)
import Data.Ord (comparing)
import Data.Void (Void, absurd)
import Prelude hiding (id, (.))

-- | A one-dimensional chain of four states; 'Goal' is the absorbing target.
data State = S0 | S1 | S2 | Goal
  deriving stock (Eq, Show, Enum, Bounded, Ord)

-- | Move left or right; edges are clamped.
data Action = L | R
  deriving stock (Eq, Show, Enum, Bounded, Ord)

-- | Deterministic transition.
--
-- >>> step R S0
-- S1
--
-- >>> step L S0
-- S0
--
-- >>> step R S2
-- Goal
step :: Action -> State -> State
step L S0 = S0
step L S1 = S0
step L S2 = S1
step L Goal = Goal
step R S0 = S1
step R S1 = S2
step R S2 = Goal
step R Goal = Goal

-- | State reward: living penalty, goal bonus.
--
-- >>> reward S0
-- -1.0
--
-- >>> reward Goal
-- 10.0
reward :: State -> Double
reward Goal = 10
reward _ = -1

-- | One-step Bellman backup for a fixed deterministic policy.
bellmanPolicy :: Double -> Action -> (State -> Double) -> State -> Double
bellmanPolicy gamma a v s = reward s + gamma * v (step a s)

-- | One-step Bellman optimality backup.
bellmanOpt :: Double -> (State -> Double) -> State -> Double
bellmanOpt gamma v s = max (bellmanPolicy gamma L v s) (bellmanPolicy gamma R v s)

-- | Finite-horizon value iteration from the zero value function.
--
-- >>> valueIter 0 0.9 S0
-- 0.0
--
-- >>> valueIter 1 0.9 S0
-- -1.0
--
-- >>> valueIter 2 0.9 S0
-- -1.9
valueIter :: Int -> Double -> State -> Double
valueIter 0 _ _ = 0
valueIter n gamma s = bellmanOpt gamma (valueIter (n - 1) gamma) s

-- | Greedy policy with respect to a value function.
optimalPolicy :: Double -> (State -> Double) -> State -> Action
optimalPolicy gamma v s = maximumBy (comparing (\a -> bellmanPolicy gamma a v s)) [L, R]

-- ---------------------------------------------------------------------------
-- Prob-composition view
-- ---------------------------------------------------------------------------

-- | State-dependent score modality. Not exported by 'Circuit.Prob' because it
-- leaks the input into the scalar map; useful for RL rewards.
scoreBy :: (a -> r -> r) -> Prob (->) r a a
scoreBy f = Prob $ \k (x, a) -> f a (k (x, a))

-- | Transition as a Prob morphism.
transP :: Action -> Prob (->) Double State State
transP = embed . step

-- | Reward as a state-dependent score modality.
rewardP :: Prob (->) Double State State
rewardP = scoreBy (\s v -> reward s + v)

-- | Discount as a scalar score modality.
discountP :: Double -> Prob (->) Double State State
discountP gamma = score (gamma *)

-- | Bellman backup for a fixed action, expressed as three Prob morphisms:
-- reward, then discount, then transition. The contravariant composition in
-- 'Prob' reads right-to-left on continuations, so the written order is the
-- operational order.
bellmanP :: Double -> Action -> Prob (->) Double State State
bellmanP gamma a = transP a . discountP gamma . rewardP

-- | Apply a Prob Bellman backup to a value function at a state.
backupP :: Double -> Action -> (State -> Double) -> State -> Double
backupP gamma a v s =
  runProb (bellmanP gamma a) (\(_, s') -> v s') ((), s)

-- ---------------------------------------------------------------------------
-- Discounted-return oracle
-- ---------------------------------------------------------------------------

-- | N-step discounted return via Prob composition.
--
-- Composes 'bellmanP gamma a' @n@ times via the 'Category' instance, then
-- applies the result to a zero continuation.  By the laws of 'Prob' 'Category'
-- composition, this is the n-step Bellman backup: reward on each step,
-- discounted and summed.
--
-- >>> discountedReturn 0.5 R 4 S0
-- -0.5
discountedReturn :: Double -> Action -> Int -> State -> Double
discountedReturn gamma a n s =
  let chainP = foldr (.) id (replicate n (bellmanP gamma a))
   in runProb chainP (\((), _) -> 0) ((), s)

-- | Closed-form discounted return on a deterministic chain.
--
-- /Σ_{t=0}^{n-1}/ /γ/^t · reward(step^t(a, s)).
-- All terms are exact in 'Double' when /γ/ is a dyadic rational (e.g. 0.5)
-- and rewards are integers.
--
-- >>> closedFormReturn 0.5 R 4 S0
-- -0.5
closedFormReturn :: Double -> Action -> Int -> State -> Double
closedFormReturn gamma a n s =
  sum $ zipWith (*) (map reward states) (iterate (gamma *) 1)
  where
    states = take n $ iterate (step a) s

-- ---------------------------------------------------------------------------
-- Tropical / shortest-path row
-- ---------------------------------------------------------------------------

-- | Min-plus tropical semiring over 'Double'.
newtype Tropical = Tropical {getTropical :: Double}
  deriving stock (Eq, Ord, Show)

-- | Tropical addition: minimum.
tAdd :: Tropical -> Tropical -> Tropical
tAdd (Tropical a) (Tropical b) = Tropical (min a b)

-- | Tropical multiplication: ordinary addition.
tMul :: Tropical -> Tropical -> Tropical
tMul (Tropical a) (Tropical b) = Tropical (a + b)

-- | Cost of each action: 0 at the goal, 1 elsewhere.
cost :: State -> Tropical
cost Goal = Tropical 0
cost _ = Tropical 1

-- | One-step tropical Bellman backup (shortest path to goal).
bellmanTropical :: (State -> Tropical) -> State -> Tropical
bellmanTropical v s =
  foldl1 tAdd [cost s `tMul` v (step a s) | a <- [L, R]]

-- | Finite-horizon shortest-path cost to goal.
--
-- >>> getTropical (shortestPath 0 S0)
-- Infinity
--
-- >>> getTropical (shortestPath 1 S0)
-- Infinity
--
-- >>> getTropical (shortestPath 2 S0)
-- Infinity
--
-- >>> getTropical (shortestPath 3 S0)
-- 3.0
--
-- >>> getTropical (shortestPath 3 Goal)
-- 0.0
shortestPath :: Int -> State -> Tropical
shortestPath 0 Goal = Tropical 0
shortestPath 0 _ = Tropical (1 / 0)
shortestPath n s = bellmanTropical (shortestPath (n - 1)) s

-- ---------------------------------------------------------------------------
-- System (Prob) view: controlled MDP
-- ---------------------------------------------------------------------------

-- | Local semiring class for the expectation runner (mirrors the kepler
-- executable, kept local to avoid substrate churn).
class Semiring r where
  sAdd :: r -> r -> r
  sMul :: r -> r -> r
  sZero :: r
  sOne :: r

instance Semiring Double where
  sAdd = (+)
  sMul = (*)
  sZero = 0
  sOne = 1

-- | Step a finite-state stochastic Moore machine by expectation, exactly as in
-- the circuits keystone, but specialised to 'Mono i o' with full state
-- observation.
expectSystem ::
  (Eq s, Semiring r) =>
  [s] ->
  System (Prob (->) r) s (Mono i o) ->
  [i] ->
  (s -> r) ->
  s ->
  r
expectSystem states sys is q s0 =
  foldl' sAdd sZero [q s `sMul` distFinal s | s <- states]
  where
    distFinal = foldl' step initDist is
    initDist s = if s == s0 then sOne else sZero
    step dist i s' =
      foldl' sAdd sZero [dist s `sMul` pTrans s i s' | s <- states]
    pTrans s i s' =
      runProb
        (runSystem sys)
        (\((), (s'', _)) -> if s' == s'' then sOne else sZero)
        ((), (s, monoIn i))

-- | The gridworld as a controlled stochastic Moore machine.
--
-- Input: action ('L' or 'R'). Output: full state observation.
gridSystem :: System (Prob (->) Double) State (Mono Action State)
gridSystem = system $ Prob $ \k (x, (s, d)) ->
  let s' = step (monoDir d) s
   in k (x, (s', (s', ())))

-- ---------------------------------------------------------------------------
-- MDP and POMDP polynomial shapes
-- ---------------------------------------------------------------------------

-- | MDP interface: action in, next-state and reward out.
--
-- This matches the instance-table claim that the MDP row uses
-- @Mono a (s', r)@.  The reward is pinned on the current state to match
-- 'bellmanSystem' / 'bellmanOpt'.
mdpSystem :: System (Prob (->) Double) State (Mono Action (State, Double))
mdpSystem = system $ Prob $ \k (x, (s, d)) ->
  let a = monoDir d
      s' = step a s
   in k (x, (s', ((s', reward s), ())))

-- | Check one deterministic MDP step by continuation.
mdpCheck :: Action -> State -> State -> Double -> Bool
mdpCheck a s expectedS' expectedR =
  runProb (runSystem mdpSystem) checkCont ((), (s, monoIn a)) == 1.0
  where
    checkCont (_, (_sNext, ((s'', r), ()))) =
      if s'' == expectedS' && r == expectedR then 1.0 else 0.0

-- | POMDP observation: coarse location, not the true state.
data Observation = Far | Near | AtGoal
  deriving stock (Eq, Show, Enum, Bounded, Ord)

-- | Coarse observation function.
observe :: State -> Observation
observe S0 = Far
observe S1 = Near
observe S2 = Near
observe Goal = AtGoal

-- | POMDP interface: hidden state carried as a 'Const' position, external loop
-- is action in / observation out.
--
-- This matches the instance-table claim that the POMDP row uses a state-hiding
-- @Prod (Const s) (Mono a o)@.  The @Const s@ position exposes the hidden
-- state as output but supplies no direction, so the external agent cannot feed
-- it back as input.
pomdpSystem :: System (Prob (->) Double) State (Prod (Const State) (Mono Action Observation))
pomdpSystem = system $ Prob $ \k (x, (s, d)) ->
  case d of
    Left v -> absurd v
    Right dMono -> case dMono of
      Left v -> absurd v
      Right a ->
        let s' = step a s
            o = observe s'
         in k (x, (s', (s', (o, ()))))

-- | Check one deterministic POMDP step by continuation.
pomdpCheck :: Action -> State -> State -> Observation -> Bool
pomdpCheck a s expectedS' expectedO =
  runProb (runSystem pomdpSystem) checkCont ((), (s, Right (monoIn a))) == 1.0
  where
    checkCont (_, (_sNext, (hidden, (obs, ())))) =
      if hidden == expectedS' && obs == expectedO then 1.0 else 0.0

-- | One-step Bellman optimality backup via 'System (Prob)'.
--
-- Reward is pinned on the /current/ state (matching 'bellmanOpt'); the System
-- runner computes the expected discounted future value of the next state.
bellmanSystem :: Double -> (State -> Double) -> State -> Double
bellmanSystem gamma v s =
  reward s
    + gamma
      * maximum
        [ expectSystem
            [S0, S1, S2, Goal]
            gridSystem
            [a]
            v
            s
        | a <- [L, R]
        ]

-- | Finite-horizon value iteration using the 'System' runner.
valueIterSystem :: Int -> Double -> State -> Double
valueIterSystem 0 _ _ = 0
valueIterSystem n gamma s = bellmanSystem gamma (valueIterSystem (n - 1) gamma) s
