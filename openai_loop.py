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
import openai
import base64

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

# --- Baseline comparison parameters ---
SQUARE_SIZE = 80  # px for warped board

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

def get_piece_map_from_frame(frame, corners):
    temp_path = "screenshots/annotated_board.jpg"
    cv2.imwrite(temp_path, frame)
    result = CLIENT.infer(temp_path, model_id=MODEL_ID)
    piece_map = {}
    for pred in result.get('predictions', []):
        x = pred['x']
        y = pred['y']
        class_name = pred['class']
        square = get_square_from_pixel(x, y, corners)
        if square and class_name in PIECE_TO_FEN:
            piece_map[square] = PIECE_TO_FEN[class_name]
    return piece_map

# Helper to get all square names in order (a8-h8, a7-h7, ..., a1-h1)
SQUARE_NAMES = [chr(ord('a')+f)+str(r) for r in range(8,0,-1) for f in range(8)]

def infer_move_from_piece_maps(before, after, board):
    # Find squares that changed
    changes = []
    for sq in SQUARE_NAMES:
        if before.get(sq) != after.get(sq):
            changes.append(sq)
    # Simple case: one piece moved from one square to another
    if len(changes) == 2:
        from_sq, to_sq = None, None
        for sq in changes:
            if before.get(sq) and not after.get(sq):
                from_sq = sq
            elif not before.get(sq) and after.get(sq):
                to_sq = sq
        if from_sq and to_sq:
            move = chess.Move.from_uci(from_sq + to_sq)
            if move in board.legal_moves:
                return move
    # Handle captures, promotions, castling, etc. (basic version)
    # Try all legal moves and see which one matches the after map
    for move in board.legal_moves:
        temp_board = board.copy()
        temp_board.push(move)
        temp_map = {}
        for sq in SQUARE_NAMES:
            piece = temp_board.piece_at(chess.parse_square(sq))
            if piece:
                temp_map[sq] = piece.symbol() if piece.color else piece.symbol().lower()
        # Compare temp_map to after
        match = True
        for sq in SQUARE_NAMES:
            if temp_map.get(sq) != after.get(sq):
                match = False
                break
        if match:
            return move
    return None

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

def extract_squares_from_board(board_image):
    """Extract individual squares from the warped board image."""
    squares = []
    for row in range(8):
        square_row = []
        for col in range(8):
            y1 = row * SQUARE_SIZE
            y2 = (row + 1) * SQUARE_SIZE
            x1 = col * SQUARE_SIZE
            x2 = (col + 1) * SQUARE_SIZE
            square = board_image[y1:y2, x1:x2]
            square_row.append(square)
        squares.append(square_row)
    return squares

def get_board_perspective(frame, corners):
    """Extract and warp the board to a square perspective."""
    dst = np.array([[0, 0], [8*SQUARE_SIZE, 0], [8*SQUARE_SIZE, 8*SQUARE_SIZE], [0, 8*SQUARE_SIZE]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)
    warped = cv2.warpPerspective(frame, matrix, (8*SQUARE_SIZE, 8*SQUARE_SIZE))
    return warped

def capture_baseline_squares(board_image, starting_board):
    """Capture baseline images and stats for each square."""
    squares = extract_squares_from_board(board_image)
    baseline = {}
    for row in range(8):
        for col in range(8):
            square_key = f"{row}_{col}"
            piece = starting_board[row][col]
            gray = cv2.cvtColor(squares[row][col], cv2.COLOR_BGR2GRAY)
            baseline[square_key] = {
                'piece': piece,
                'avg_brightness': np.mean(gray),
                'std_brightness': np.std(gray),
                'avg_color': np.mean(squares[row][col])
            }
    return baseline

def detect_piece_simple(square_image, baseline_stats, row, col):
    square_key = f"{row}_{col}"
    if square_key not in baseline_stats:
        return '.'
    baseline = baseline_stats[square_key]
    expected_piece = baseline['piece']
    gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
    current_avg = np.mean(gray)
    current_std = np.std(gray)
    brightness_diff = abs(current_avg - baseline['avg_brightness'])
    std_diff = abs(current_std - baseline['std_brightness'])
    if brightness_diff < 15 and std_diff < 10:
        return expected_piece
    if current_std < 15:
        return '.'
    is_white_piece = current_avg > 120
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    if edge_density < 0.1:
        return '.'
    elif edge_density < 0.25:
        return 'P' if is_white_piece else 'p'
    elif edge_density < 0.35:
        return 'R' if is_white_piece else 'r'
    elif edge_density < 0.45:
        return 'B' if is_white_piece else 'b'
    elif edge_density < 0.55:
        return 'N' if is_white_piece else 'n'
    elif edge_density < 0.65:
        return 'Q' if is_white_piece else 'q'
    else:
        return 'K' if is_white_piece else 'k'

def board_array_to_fen(board_array):
    fen_rows = []
    for row in board_array:
        fen_row = ""
        empty_count = 0
        for piece in row:
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    board_fen = '/'.join(fen_rows)
    full_fen = f"{board_fen} w KQkq - 0 1"
    return full_fen

def fen_to_board_array(fen_board):
    board = []
    rows = fen_board.split('/')
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    board_row.append('.')
            else:
                board_row.append(char)
        board.append(board_row)
    return np.array(board)

def get_fen_from_openai(image_path):
    """Send the board image to OpenAI Vision and extract the FEN string."""
    prompt = (
        "What is the FEN for this chessboard? Only return the FEN string. "
        "If the board is not fully visible or unclear, return your best guess."
    )
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a chessboard FEN detector."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]},
            ],
            max_tokens=100,
        )
    # Extract FEN from response
    fen = response.choices[0].message.content.strip().split()[0]
    return fen

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
    # --- OpenAI Vision FEN detection ---
    image_path = 'screenshots/board_for_openai.jpg'
    cv2.imwrite(image_path, frame)
    detected_fen = get_fen_from_openai(image_path)
    print(f"[OpenAI] Detected FEN: {detected_fen}")
    try:
        board = chess.Board(detected_fen)
    except Exception as e:
        print(f"[ERROR] Could not parse FEN: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return
    save_annotated_screenshot(frame, corners, "init")
    last_fen = detected_fen
    move_history = []
    print("\nMonitoring board state. Press 'q' to quit.")
    try:
        while not board.is_game_over():
            # Wait for user to press SPACE to capture the next board state
            print("Press SPACE to capture the next board state (after a move). Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
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
            # Save and send to OpenAI
            image_path = 'screenshots/board_for_openai.jpg'
            cv2.imwrite(image_path, frame)
            detected_fen = get_fen_from_openai(image_path)
            print(f"[OpenAI] Detected FEN: {detected_fen}")
            if detected_fen == last_fen:
                print("[NO CHANGE] No move detected. Waiting for next sample...")
                continue
            try:
                new_board = chess.Board(detected_fen)
                move = None
                for m in board.legal_moves:
                    temp_board = board.copy()
                    temp_board.push(m)
                    if temp_board.board_fen() == new_board.board_fen():
                        move = m
                        break
                if move is None:
                    print("Couldn't detect a valid move. Waiting for next sample...")
                    last_fen = detected_fen
                    continue
                board.push(move)
                move_history.append(move)
                print(f"[MOVE] You played: {move}")
                print(board)
                last_fen = detected_fen
                save_annotated_screenshot(frame, corners, f"user_{move}")
                # --- AI move suggestion ---
                fen = board.fen()
                reply_move, eval_info = get_best_move(fen)
                if reply_move is None:
                    print("Engine resigns or no move found.")
                    break
                print(f"🤖 My move: {reply_move} | Eval: {eval_info}")
                score = eval_info.get("value", 0)
                smack = generate_trash_talk(score)
                print(f"🗯️  {smack}")
                speak_text(f"My move is {reply_move}. {smack}")
                print("Please make my move on the board, then press SPACE to continue.")
                # Wait for user to make the AI's move and press SPACE
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
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
                # After user confirms, update the board state for the AI's move
                board.push(chess.Move.from_uci(reply_move))
                move_history.append(chess.Move.from_uci(reply_move))
                ret, frame = cap.read()
                if ret:
                    save_annotated_screenshot(frame, corners, f"engine_{reply_move}")
                last_fen = board.fen()
            except Exception as e:
                print(f"Error updating board: {e}")
                import traceback
                traceback.print_exc()
                last_fen = detected_fen
                continue
        print("\n🏁 Game Over:", board.result())
    finally:
        # Always save the game, even if not completed
        result = board.result() if board.is_game_over() else "*"
        save_game_pgn(move_history, result)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
