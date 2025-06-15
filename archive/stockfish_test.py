from stockfish import Stockfish

stockfish = Stockfish(path="/opt/homebrew/bin/stockfish")
stockfish.set_position(["e2e4", "e7e5", "g1f3"])
print("Best move:", stockfish.get_best_move())

print(stockfish.get_fen_position())
print(stockfish.get_board_visual())
print(stockfish.get_top_moves(3))
