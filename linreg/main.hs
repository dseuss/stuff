import qualified CSV
import qualified Linreg


main = do
     putStrLn "LINREG -- linear regression in one variable"

     putStrLn "Reading data..."
     contents <- readFile "data.csv"

     {- TODO Clean this up a little -}
     let order = 1
         initial = replicate (order + 1) 0 :: [Double]
         result = Linreg.fit (CSV.parse contents) initial

     print (result !! 1000)
     putStrLn "Done."


-- Creates polynomial function: poly m x = f(x) = m_0 + m_1*x + m_2*x^2 + ...
poly :: (Num a) => [a] -> (a -> a)
poly m x = sum $ map (x^) [0..order]
   where order = length m - 1


