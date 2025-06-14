from inference_sdk import InferenceHTTPClient
import json
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import chess

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# result = CLIENT.infer("assets/chess_board_2.jpg", model_id="chess-piece-detection-5ipnt/3")
# print(json.dumps(result, indent=4))

# Estimated pixel coordinates for the four corners of the board in chess_board_2.jpg
# Order: a8, h8, h1, a1
BOARD_CORNERS = [
    (1540, 3600),   # a8 (top-left)
    (4560, 3500),  # h8 (top-right)
    (5000, 80), # h1 (bottom-right)
    (710, 410),  # a1 (bottom-left)
]

# Helper: map (x, y) to board square
def get_square_from_pixel(x, y, corners=BOARD_CORNERS):
    # Perspective transform to map pixel to board coordinates
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0, 0], [7, 0], [7, 7], [0, 7]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    px = np.array([[x, y]], dtype=np.float32)
    px = np.array([px])
    board_xy = cv2.perspectiveTransform(px, M)[0][0]
    file = int(round(board_xy[0]))
    rank = int(round(board_xy[1]))
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chr(ord('a') + file) + str(8 - rank)
    return None

# Helper: build FEN from piece positions
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

# Map model class names to FEN letters
PIECE_TO_FEN = {
    'white-pawn': 'P', 'white-rook': 'R', 'white-knight': 'N', 'white-bishop': 'B', 'white-queen': 'Q', 'white-king': 'K',
    'black-pawn': 'p', 'black-rook': 'r', 'black-knight': 'n', 'black-bishop': 'b', 'black-queen': 'q', 'black-king': 'k',
}

# Use the new image for inference
result = CLIENT.infer("assets/chess_board_2.jpg", model_id="chess-piece-detection-5ipnt/3")

# Build piece map
piece_map = {}
for pred in result.get('predictions', []):
    x = pred['x']
    y = pred['y']
    class_name = pred['class']
    square = get_square_from_pixel(x, y)
    if square and class_name in PIECE_TO_FEN:
        piece_map[square] = PIECE_TO_FEN[class_name]

fen = build_fen(piece_map)
print("FEN:", fen)

# Visual representation using python-chess
board = chess.Board(fen)
print(board)
