{-# LANGUAGE ScopedTypeVariables #-}

module DecisionTree where

import qualified Data.Vector as V
import qualified Data.Vector.Generic as VG
import qualified Data.List as List
import qualified Data.List.Extras.Argmax as Argmax
import qualified Statistics.Sample as S


-- Returns the list of unique elements of x --
uniqueElems :: (Eq a) => V.Vector a -> [a]
uniqueElems x = List.nub $ V.toList x

-- Divide two Integral values --
intDiv :: (Integral a, Fractional b) => a -> a -> b
intDiv x y = fromIntegral x / fromIntegral y


-- Returns the sample probability mass function of `samples` P(X = x)
samplePMF :: (Eq a) => V.Vector a -> a -> Double
samplePMF samples x = V.length (V.filter (== x) samples) `intDiv` V.length samples


-- Returns the entropy from the samples `y`. In terms of the sample
-- frequencies p_j (for the j-the value) it is defined by
--                 H(Y) =  -\sum_j p_j * log_2 p_j
entropy :: forall a. (Eq a) => V.Vector a -> Double
entropy y = (-1.0) * sum (map partialEntropy (uniqueElems y))
    where partialEntropy :: a -> Double
          partialEntropy i = pr i * logBase 2 (pr i)

          pr :: a -> Double
          pr = samplePMF y


-- Calculates the "information gain" of the sample `y` when we subdivide
-- it into groups according to the labels given by `lab`
infoGain :: forall a b. (Eq a, Eq b) => V.Vector a -> V.Vector b -> Double
infoGain y lab
    | V.length y /= V.length lab = error $ "DecisionTree.hs:infoGain " ++
                                           "Dimensions do not match."
    | otherwise = (-1.0) * sum (map partialInfoGain subsets)
        where partialInfoGain :: V.Vector a -> Double
              partialInfoGain a = pr_a * entropy a
                  where pr_a = V.length a `intDiv` V.length y

              subsets :: [V.Vector a]
              subsets = map elemsWithLabel (uniqueElems lab)

              elemsWithLabel :: b -> V.Vector a
              elemsWithLabel a = VG.ifilter (\ i _ -> lab V.! i == a) y


fitTree :: forall a b. (Ord a, Eq b) => [V.Vector a] -> V.Vector b -> DNode [a] [Double]
fitTree x y = Decision ((decF bestP) . bestP)  (Result "left") (Result "right")
    where decF :: [a] -> a -> a -> Bool
          decF p z = z < S.mean (p x)

          projectors :: [[a] -> a]
          projectors = [\z -> (z !! n) | n <- [0..(length x - 1)]]
          bestP :: [a] -> a
          bestP = Argmax.argmax (\ p -> (infoGain y $ V.map (decF p) (p x))) projectors


-- a: experiment to decide on
-- b: type of result i.e. classification or classification probability
data DNode a b = Result b
               | Decision (a -> Bool) (DNode a b) (DNode a b)


-- Apply the decision tree to a single experiment
decide :: DNode a b -> a -> b
decide (Result p) = p
decide (Decision df tbranch fbranch) x =
        if df x
            then decide tbranch x
            else decide fbranch x
