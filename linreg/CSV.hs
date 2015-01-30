module CSV
( parse
) where


parse :: (Read a) => String -> [[a]]
parse content = convert $ raw content
   where raw :: String -> [[String]]
         convert :: (Read a) => [[String]] -> [[a]]
         raw input = map (splitListBy ',') (lines input)
         convert input = map (map read) input :: Read a => [[a]]


splitListBy :: (Eq a) => a -> [a] -> [[a]]
splitListBy _ [] = []
splitListBy delim l
   | head l == delim = splitListBy delim (tail l)
   | otherwise       = l1 : splitListBy delim l2
   where (l1, l2) = break (== delim) l
