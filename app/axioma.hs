module Main where

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
                (\s ->
                   approx (backupP 0.9 L v s) (bellmanPolicy 0.9 L v s)
                     && approx (backupP 0.9 R v s) (bellmanPolicy 0.9 R v s))
                [S0, S1, S2, Goal],
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
            [S0, S1, S2, Goal]
      ]
  if and results
    then putStrLn "\nAll tests passed."
    else error "Some tests failed."
