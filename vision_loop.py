import cv2
import numpy as np
import os
import time
import chess
from stockfish import Stockfish
from elevenlabs_tts import speak_text
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import datetime
import chess.pgn

# STATUS:
# - Vision mode is working, but the board detection is not perfect.
# - Misidentifies pawns as bishops and doesn't see all the pieces.

# Load environment variables for Roboflow
load_dotenv()

# --- Roboflow Setup ---
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)
MODEL_ID = "chess-piece-detection-5ipnt/3"

# --- Stockfish Setup ---
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Adjust if needed
engine = Stockfish(path=STOCKFISH_PATH)

# --- Board Corners (manually set for now, or could be detected) ---
# Order: a8, h8, h1, a1 (top-left, top-right, bottom-right, bottom-left)
BOARD_CORNERS = [
    (1640, 860),   # a8 (top-left)
    (4430, 950),   # h8 (top-right)
    (4730, 3990),  # h1 (bottom-right)
    (950, 3710),   # a1 (bottom-left)
]

PIECE_TO_FEN = {
    'white-pawn': 'P', 'white-rook': 'R', 'white-knight': 'N', 'white-bishop': 'B', 'white-queen': 'Q', 'white-king': 'K',
    'black-pawn': 'p', 'black-rook': 'r', 'black-knight': 'n', 'black-bishop': 'b', 'black-queen': 'q', 'black-king': 'k',
}

def manual_corner_selection(cap):
    print("Manual corner selection - click corners in order: top-left (a8), top-right (h8), bottom-right (h1), bottom-left (a1)")
    corners = []
    window_name = 'Select Board Corners'
    selected = {'corners': []}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected['corners']) < 4:
            selected['corners'].append([x, y])
            print(f"Corner {len(selected['corners'])}: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Draw selected corners
        for i, corner in enumerate(selected['corners']):
            cv2.circle(frame, tuple(corner), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (corner[0]+10, corner[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Click corner {len(selected['corners'])+1}/4",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        if len(selected['corners']) == 4:
            cv2.destroyWindow(window_name)
            return np.array(selected['corners'], dtype=np.float32)

def get_square_from_pixel(x, y, corners):
    # Use the same perspective transform as the grid overlay
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0, 0], [800, 0], [800, 800], [0, 800]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    px = np.array([[[x, y]]], dtype=np.float32)  # shape (1, 1, 2)
    board_xy = cv2.perspectiveTransform(px, M)[0][0]
    file = int(board_xy[0] // 100)
    rank = int(board_xy[1] // 100)
    if 0 <= file <= 7 and 0 <= rank <= 7:
        # Convert to chess notation: file (a-h), rank (8-1)
        return chr(ord('a') + file) + str(8 - rank)
    return None

def build_fen(piece_map):
    fen = ''
    for rank in range(8, 0, -1):
        empty = 0
        for file in range(8):
            sq = chr(ord('a') + file) + str(rank)
            piece = piece_map.get(sq)
            if piece:
                if empty:
                    fen += str(empty)
                    empty = 0
                fen += piece
            else:
                empty += 1
        if empty:
            fen += str(empty)
        if rank > 1:
            fen += '/'
    return fen

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

def detect_board_fen_from_frame(frame, corners):
    temp_path = "screenshots/annotated_board.jpg"
    cv2.imwrite(temp_path, frame)
    result = CLIENT.infer(temp_path, model_id=MODEL_ID)
    piece_map = {}
    for pred in result.get('predictions', []):
        x = pred['x']
        y = pred['y']
        class_name = pred['class']
        print(f"Detected: {class_name} at ({x:.1f}, {y:.1f})")
        square = get_square_from_pixel(x, y, corners)
        if square and class_name in PIECE_TO_FEN:
            print(f"Adding {class_name} to {square}")
            piece_map[square] = PIECE_TO_FEN[class_name]
    print("Piece map (square: piece):", piece_map)
    fen = build_fen(piece_map)
    fen_full = f"{fen} w KQkq - 0 1"
    return fen_full

def save_annotated_screenshot(frame, corners, move_label):
    annotated = frame.copy()
    # Draw corners
    for i, corner in enumerate(corners):
        cv2.circle(annotated, tuple(map(int, corner)), 8, (0, 255, 0), -1)
        cv2.putText(annotated, str(i+1), (int(corner[0])+10, int(corner[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Draw lines between corners
    for i in range(4):
        pt1 = tuple(map(int, corners[i]))
        pt2 = tuple(map(int, corners[(i+1)%4]))
        cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)
    # Perspective transform for grid overlay
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0, 0], [8*100, 0], [8*100, 8*100], [0, 8*100]
    ], dtype=np.float32)  # 800x800 px board for grid
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Draw grid lines in rectified space, then map back
    grid_size = 8
    board_px = 8*100
    for i in range(grid_size+1):
        # Vertical lines
        pt1 = np.array([[i*100, 0]], dtype=np.float32)
        pt2 = np.array([[i*100, board_px]], dtype=np.float32)
        pts = np.array([pt1, pt2]).reshape(-1,1,2)
        pts = cv2.perspectiveTransform(pts, Minv)
        cv2.line(annotated, tuple(pts[0][0].astype(int)), tuple(pts[1][0].astype(int)), (0,255,0), 2)
        # Horizontal lines
        pt3 = np.array([[0, i*100]], dtype=np.float32)
        pt4 = np.array([[board_px, i*100]], dtype=np.float32)
        pts2 = np.array([pt3, pt4]).reshape(-1,1,2)
        pts2 = cv2.perspectiveTransform(pts2, Minv)
        cv2.line(annotated, tuple(pts2[0][0].astype(int)), tuple(pts2[1][0].astype(int)), (255,0,0), 2)
    # Add timestamp and move label
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    label = f"{move_label}-{timestamp}"
    cv2.putText(annotated, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    # Ensure screenshots folder exists
    os.makedirs('screenshots', exist_ok=True)
    filename = f"screenshots/{label}.jpg"
    cv2.imwrite(filename, annotated)
    print(f"Saved screenshot: {filename}")

def save_game_pgn(move_history, game_result):
    os.makedirs('games', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"games/game_{timestamp}.pgn"
    game = chess.pgn.Game()
    node = game
    for move in move_history:
        node = node.add_variation(move)
    game.headers["Result"] = game_result
    with open(filename, 'w') as f:
        print(game, file=f)
    print(f"Saved game PGN: {filename}")

def main():
    print("🧠 Cagnus Marlsen Vision Mode: Set up your board, then select the four corners.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    # Manual corner selection
    corners = manual_corner_selection(cap)
    if corners is None or len(corners) != 4:
        print("Corner selection cancelled or failed.")
        cap.release()
        cv2.destroyAllWindows()
        return
    print(f"Selected corners: {corners.tolist()}")
    print("Press SPACE to capture the initial board state.")
    # Wait for user to press SPACE to start
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Draw corners on frame
        for i, corner in enumerate(corners):
            cv2.circle(frame, tuple(map(int, corner)), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (int(corner[0])+10, int(corner[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Chess Vision', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    print("Detecting initial board state...")
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture initial frame.")
        return
    fen = detect_board_fen_from_frame(frame, corners)
    print(f"Initial FEN: {fen}")
    # Save annotated screenshot with grid at initiation
    save_annotated_screenshot(frame, corners, "init")
    board = chess.Board(fen)
    print(board)
    last_fen = fen
    move_history = []
    while not board.is_game_over():
        print("Waiting for your move... (update the board, then press SPACE)")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            # Draw corners on frame
            for i, corner in enumerate(corners):
                cv2.circle(frame, tuple(map(int, corner)), 8, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (int(corner[0])+10, int(corner[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Chess Vision', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        fen = detect_board_fen_from_frame(frame, corners)
        if fen == last_fen:
            print("No change detected. Try again.")
            continue
        try:
            new_board = chess.Board(fen)
            move = None
            for m in board.legal_moves:
                temp_board = board.copy()
                temp_board.push(m)
                if temp_board.board_fen() == new_board.board_fen():
                    move = m
                    break
            if move is None:
                print("Couldn't detect a valid move. Try again.")
                continue
            board.push(move)
            move_history.append(move)
            print(f"You played: {move}")
            print(board)
            last_fen = fen
            # Save screenshot after user move
            save_annotated_screenshot(frame, corners, f"user_{move}")
        except Exception as e:
            print(f"Error updating board: {e}")
            continue
        fen = board.fen()
        reply_move, eval_info = get_best_move(fen)
        if reply_move is None:
            print("Engine resigns or no move found.")
            break
        # Capture frame for engine move
        ret, frame = cap.read()
        if ret:
            save_annotated_screenshot(frame, corners, f"engine_{reply_move}")
        board.push(chess.Move.from_uci(reply_move))
        move_history.append(chess.Move.from_uci(reply_move))
        print(f"🤖 My move: {reply_move} | Eval: {eval_info}")
        print(board)
        score = eval_info.get("value", 0)
        smack = generate_trash_talk(score)
        print(f"🗯️  {smack}")
        speak_text(smack)
        last_fen = board.fen()
    print("\n🏁 Game Over:", board.result())
    save_game_pgn(move_history, board.result())
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
