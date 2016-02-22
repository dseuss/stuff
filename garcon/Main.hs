-- TODO Better Logging
module Main (main) where


import System.Environment
import System.IO
import Network.Socket
import Debug.Trace


data CmdArgs = CmdArgs {
          socketNr :: Int
          }
    deriving(Show)


parseArgs :: [String] -> CmdArgs
parseArgs [] = CmdArgs { socketNr = 4242 }
parseArgs (arg:_) = CmdArgs{ socketNr = read arg }


listenLoop :: Socket -> IO()
listenLoop sock = do
        connection <- accept sock
        print $ "[CONNECTED] " ++ (show connection)
        listenLoop sock


main :: IO ()
main = do
        -- parse command line arguments
        rawArgs <- getArgs
        let cmdArgs = parseArgs rawArgs

        -- setup socket
        sock <- socket AF_INET Stream 0
        setSocketOption sock ReuseAddr 2
        let portNr = fromIntegral $ socketNr cmdArgs
            hostAddr = iNADDR_ANY
        print $ "[SETUP] On " ++ (show hostAddr) ++ ":" ++ show portNr
        bind sock (SockAddrInet portNr iNADDR_ANY)
        listen sock 1
        listenLoop sock
