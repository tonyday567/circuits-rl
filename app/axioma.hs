module Main where

import Circuit.RL.Estimator
import Circuit.RL.GridWorld
import Prelude hiding (id, (.))

approx :: Double -> Double -> Bool
approx x y = abs (x - y) < 1e-9

check :: String -> Bool -> IO Bool
check name ok = do
  putStrLn $ (if ok then "PASS " else "FAIL ") ++ name
  pure ok

main :: IO ()
main = do
  results <-
    sequence
      [ check "GridWorld step deterministic" $
          step R S0 == S1 && step L S0 == S0 && step R S2 == Goal,
        check "GridWorld reward" $
          reward S0 == -1 && reward Goal == 10,
        check "GridWorld valueIter 0 is zero" $
          valueIter 0 0.9 S0 == 0,
        check "GridWorld valueIter 1" $
          valueIter 1 0.9 S0 == -1,
        check "GridWorld valueIter 2" $
          approx (valueIter 2 0.9 S0) (-1.9),
        check "GridWorld optimal policy points right" $
          optimalPolicy 0.9 (valueIter 10 0.9) S0 == R
            && optimalPolicy 0.9 (valueIter 10 0.9) S1 == R
            && optimalPolicy 0.9 (valueIter 10 0.9) S2 == R,
        check "GridWorld backupP agrees with bellmanPolicy" $
          approx
            (backupP 0.9 R (valueIter 5 0.9) S1)
            (bellmanPolicy 0.9 R (valueIter 5 0.9) S1),
        check "GridWorld Prob composition is Bellman backup" $
          let v = valueIter 4 0.9
           in all
                ( \s ->
                    approx (backupP 0.9 L v s) (bellmanPolicy 0.9 L v s)
                      && approx (backupP 0.9 R v s) (bellmanPolicy 0.9 R v s)
                )
                [S0, S1, S2, Goal],
        check "Discount oracle: 4-step from S0 exact" $
          discountedReturn 0.5 R 4 S0 == -0.5
            && closedFormReturn 0.5 R 4 S0 == -0.5,
        check "Discount oracle: Prob == closed form, all n≤5, all states" $
          all
            (\(n, s) -> discountedReturn 0.5 R n s == closedFormReturn 0.5 R n s)
            [(n, s) | n <- [1 .. 5], s <- [S0, S1, S2, Goal]],
        check "Discount oracle: hand values match closed form" $
          closedFormReturn 0.5 R 1 S0 == -1
            && closedFormReturn 0.5 R 2 S0 == -1.5
            && closedFormReturn 0.5 R 3 S0 == -1.75
            && closedFormReturn 0.5 R 4 S0 == -0.5,
        check "GridWorld tropical shortestPath 0" $
          let Tropical x = shortestPath 0 S0 in isInfinite x,
        check "GridWorld tropical shortestPath hand values" $
          let Tropical d0 = shortestPath 1 S0
              Tropical d1 = shortestPath 2 S0
              Tropical d2 = shortestPath 3 S0
              Tropical d3 = shortestPath 3 Goal
           in isInfinite d0 && isInfinite d1 && approx d2 3 && d3 == 0,
        check "GridWorld System (Prob) typechecks" $
          length [valueIterSystem 0 0.9 S0, valueIterSystem 0 0.9 Goal] == 2,
        check "GridWorld System backup matches direct VI" $
          let v = valueIter 4 0.9
           in all
                (\s -> approx (bellmanSystem 0.9 v s) (bellmanOpt 0.9 v s))
                [S0, S1, S2, Goal],
        check "GridWorld System VI matches direct VI" $
          all
            (\s -> approx (valueIterSystem 4 0.9 s) (valueIter 4 0.9 s))
            [S0, S1, S2, Goal],
        -- -----------------------------------------------------------------------
        -- W2: REINFORCE == pathwise oracle
        -- -----------------------------------------------------------------------
        check "Estimator Dual product rule" $
          getDual (Dual (2 :: Double, 3) * Dual (5, 7)) == (10, 2 * 7 + 3 * 5),
        check "REINFORCE == closed form at θ=3, σ=0.5, a_target=1" $
          reinforceGrad 3.0 0.5 1.0 == closedFormGrad 3.0 0.5 1.0,
        check "Pathwise == closed form at θ=3, σ=0.5, a_target=1" $
          pathwiseGrad 3.0 0.5 1.0 == closedFormGrad 3.0 0.5 1.0,
        check "REINFORCE == pathwise at θ=3, σ=0.5, a_target=1" $
          reinforceGrad 3.0 0.5 1.0 == pathwiseGrad 3.0 0.5 1.0,
        check "Closed form exact: ∇J(θ=3, a_target=1) = -4" $
          closedFormGrad 3.0 0.5 1.0 == -4.0,
        check "REINFORCE / closed form sweep over θ ∈ [0, 10]" $
          all
            ( \theta ->
                reinforceGrad theta 0.5 1.0 == closedFormGrad theta 0.5 1.0
            )
            [0, 0.5, 1, 2, 3, 5, 7, 10],
        check "Pathwise / closed form sweep over θ ∈ [0, 10]" $
          all
            ( \theta ->
                pathwiseGrad theta 0.5 1.0 == closedFormGrad theta 0.5 1.0
            )
            [0, 0.5, 1, 2, 3, 5, 7, 10],
        check "REINFORCE independent of σ (as it should: both sides use σ)" $
          all
            ( \sigma ->
                reinforceGrad 3.0 sigma 1.0 == closedFormGrad 3.0 sigma 1.0
            )
            [0.1, 0.25, 0.5, 1.0, 2.0],
        check "Pathwise independent of σ" $
          all
            ( \sigma ->
                pathwiseGrad 3.0 sigma 1.0 == closedFormGrad 3.0 sigma 1.0
            )
            [0.1, 0.25, 0.5, 1.0, 2.0],
        check "REINFORCE / pathwise sweep (θ, σ) cross" $
          all
            ( \(theta, sigma) ->
                reinforceGrad theta sigma 1.0 == pathwiseGrad theta sigma 1.0
            )
            [(theta, sigma) | theta <- [0, 1, 2, 3], sigma <- [0.1, 0.5, 1.0]]
      ]
  if and results
    then putStrLn "\nAll tests passed."
    else error "Some tests failed."
