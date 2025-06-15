from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import tempfile
import base64
import chess
from diff_loop import get_board_perspective, extract_squares_from_board, detect_piece_simple, board_array_to_fen, fen_to_board_array, capture_baseline_squares

app = Flask(__name__)
CORS(app)

# In-memory session state (for demo; use a DB or cache for production)
sessions = {}

@app.route('/start-game', methods=['POST'])
def start_game():
    session_id = os.urandom(8).hex()
    sessions[session_id] = {
        'corners': None,
        'baseline_stats': None,
        'fen': chess.STARTING_FEN
    }
    return jsonify({'session_id': session_id, 'fen': chess.STARTING_FEN})

@app.route('/set-board-corners', methods=['POST'])
def set_board_corners():
    data = request.json
    session_id = data['session_id']
    corners = data['corners']  # Should be [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    sessions[session_id]['corners'] = np.array(corners, dtype=np.float32)
    return jsonify({'status': 'ok'})

@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    data = request.json
    session_id = data['session_id']
    img_b64 = data['image']
    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    corners = sessions[session_id]['corners']
    if corners is None:
        return jsonify({'error': 'Corners not set'}), 400
    warped = get_board_perspective(frame, corners)
    # For the first frame, capture baseline
    if sessions[session_id]['baseline_stats'] is None:
        starting_board = fen_to_board_array(chess.STARTING_FEN.split()[0])
        baseline_stats = capture_baseline_squares(warped, starting_board)
        sessions[session_id]['baseline_stats'] = baseline_stats
    else:
        baseline_stats = sessions[session_id]['baseline_stats']
    # Detect board state
    squares = extract_squares_from_board(warped)
    board_state = []
    for row in range(8):
        board_row = []
        for col in range(8):
            piece = detect_piece_simple(squares[row][col], baseline_stats, row, col)
            board_row.append(piece)
        board_state.append(board_row)
    fen = board_array_to_fen(board_state)
    sessions[session_id]['fen'] = fen
    return jsonify({'fen': fen})

@app.route('/get-board-state', methods=['GET'])
def get_board_state():
    session_id = request.args.get('session_id')
    fen = sessions.get(session_id, {}).get('fen')
    if fen is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'fen': fen})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
