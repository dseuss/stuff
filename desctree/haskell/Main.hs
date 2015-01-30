{-# LANGUAGE OverloadedStrings #-}

import Control.Applicative
import Control.Monad
import Data.Csv
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V


-- Data definition for the Iris dataset  ----------------------------------
data Iris = Iris
          { irisClass :: Int
          , sepalLength :: Double
          , sepalWidth :: Double
          , petalLength :: Double
          , petalWidth :: Double
          } deriving (Eq, Ord, Show, Read)

instance FromRecord Iris where
      parseRecord r = Iris <$> r .! 0 <*>
                               r .! 1 <*>
                               r .! 2 <*>
                               r .! 3 <*>
                               r .! 4

evalutateAt :: a -> (a -> b) -> b
evalutateAt x f = f x

extractFeatures :: Iris -> [Double]
extractFeatures iris = map (evalutateAt iris) [sepalLength, sepalWidth, petalLength, petalWidth]
---------------------------------------------------------------------------


-- Return the Right value of Either or raise error
fromRight :: (Show a) => Either a b -> b
fromRight (Left msg) = error $ "Main.hs:fromRight " ++ show msg
fromRight (Right x) = x


main :: IO ()
main = do
      csvData <- BL.readFile "train.csv"
      let iris = fromRight $ decode NoHeader csvData :: V.Vector Iris
      print $ V.map extractFeatures iris
      print "Done"
