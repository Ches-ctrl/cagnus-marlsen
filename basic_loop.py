# basic_loop.py
import chess
from stockfish import Stockfish

# You: Replace with your actual Stockfish binary path if needed
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# Setup
board = chess.Board()
engine = Stockfish(path=STOCKFISH_PATH)

def get_best_move(fen: str) -> tuple[str, dict]:
    engine.set_fen_position(fen)
    move = engine.get_best_move_time(100)  # 100 ms thinking time
    eval_ = engine.get_evaluation()
    return move, eval_

def generate_trash_talk(score: int) -> str:
    if score > 200:
        return "You're cooked. Might as well resign."
    elif score > 50:
        return "This is not looking good for you, pal."
    elif score > -50:
        return "Neck and neck. Let's dance."
    elif score > -200:
        return "You sure you know how the horse moves?"
    else:
        return "Are you playing blindfolded?"

# Main loop
print("🧠 Cagnus Marlsen is ready. Make your move.")
while not board.is_game_over():
    print(board)  # Print board to terminal
    user_input = input("Your move (in UCI format, e.g. e2e4): ").strip()

    if user_input == "exit":
        break

    try:
        move = chess.Move.from_uci(user_input)
        if move in board.legal_moves:
            board.push(move)
        else:
            print("❌ Illegal move. Try again.")
            continue
    except:
        print("❌ Invalid input. Use format like e2e4.")
        continue

    # Engine replies
    fen = board.fen()
    reply_move, eval_info = get_best_move(fen)
    board.push(chess.Move.from_uci(reply_move))

    # Banter
    score = eval_info.get("value", 0)
    smack = generate_trash_talk(score)
    print(f"🤖 My move: {reply_move} | Eval: {eval_info}")
    print(f"🗯️  {smack}")

print("\n🏁 Game Over:", board.result())
