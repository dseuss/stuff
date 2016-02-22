-- TODO Better Logging
-- TODO Error treatment
module Main (main) where


import Data.ByteString as BS (readFile)
import Control.Concurrent (forkIO)
import System.Environment
import System.IO
import Network.Socket
import Network.Socket.ByteString as B
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
        forkIO (handleConn connection)
        listenLoop sock


handleConn :: (Socket, SockAddr) -> IO()
handleConn (sock, _) = do
        msg <- B.recv sock 1024
        putStrLn $ "[READ] " ++ show msg
        answer <- BS.readFile "www/index.html"
        B.send sock answer
        close sock


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
