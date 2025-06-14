from stockfish import Stockfish

stockfish = Stockfish(path="/opt/homebrew/bin/stockfish")
stockfish.set_position(["e2e4", "e7e5", "g1f3"])
print("Best move:", stockfish.get_best_move())
