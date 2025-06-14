# basic_loop.py
import chess
from stockfish import Stockfish
from elevenlabs_tts import speak_text
from langgraph_agent import generate_trash_talk_with_agent

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
    print(f"FEN:{fen}")
    reply_move, eval_info = get_best_move(fen)
    board.push(chess.Move.from_uci(reply_move))

    # Banter
    score = eval_info.get("value", 0)
    smack = generate_trash_talk_with_agent(fen, score)
    print(f"🤖 My move: {reply_move} | Eval: {eval_info}")
    print(f"🗯️  {smack}")
    speak_text(smack)

print("\n🏁 Game Over:", board.result())
