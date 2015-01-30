-- FIXME Better choice than lists? FASTER? More concise?
module Linreg
( fit
) where

---------------------------------------------------------------------------
(.*) :: (Num a) => a -> [a] -> [a]
x .* y = map (x*) y

instance (Num a) => Num [a] where
      (+) = zipWithSave (+)
      (-) = zipWithSave (-)
      (*) = zipWithSave (*)
      negate = map negate
      abs = map abs
      signum = map signum
      fromInteger i = [fromInteger i]

zipWithSave :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWithSave f (x:xs) (y:ys) = (f x y):zipWithSave f xs ys
zipWithSave _ [] [] = []
zipWithSave _ _ _ = error "ERROR WITH zipWithSave"
---------------------------------------------------------------------------

-- TODO Return errors
fit :: (Fractional a) => [[a]] -> [a] -> [[a]]
fit values θinit = θinit:(gradDesc gradDescStep θinit)
      where gradDescStep θ =  -0.1 .* (costGrad values θ)


-- TODO Make stepzize variable
gradDesc :: (Fractional a) => ([a] -> [a]) -> [a] -> [[a]]
gradDesc gradDescStep θ = θ':(gradDesc gradDescStep θ')
      where θ' = θ + (gradDescStep θ)


costGrad :: (Fractional a) => [[a]] -> [a] -> [a]
costGrad values θ = normalize $ sum $ map partialCostGrad values
      where f = hypothesis θ
            partialCostGrad (y:x) = (f x - y) .* (1:x)
            normalize = map (/(fromIntegral (length values)))
            sum = foldl (+) (replicate (length θ) 0)


cost:: (Fractional a) => [[a]] -> [a] -> a
cost values θ = (sum $ map partialCost values) / (fromIntegral (2 * length values))
      where f = hypothesis θ
            partialCost (y:x) = (y - f x)^2


hypothesis :: (Num a) => [a] -> [a] -> a
hypothesis (θ0:θr) x = θ0 + θr `dot` x


dot :: (Num a) => [a] -> [a] -> a
dot x y = sum $ zipWithSave (*) x y
