from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import tempfile
import base64
import chess
from stockfish import Stockfish
from diff_loop import get_board_perspective, extract_squares_from_board, detect_piece_simple, board_array_to_fen, fen_to_board_array, capture_baseline_squares

app = Flask(__name__)
CORS(app)

# In-memory session state (for demo; use a DB or cache for production)
sessions = {}

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Adjust as needed

@app.route('/start-game', methods=['POST'])
def start_game():
    session_id = os.urandom(8).hex()
    sessions[session_id] = {
        'corners': None,
        'baseline_stats': None,
        'fen': chess.STARTING_FEN,
        'prev_fen': chess.STARTING_FEN,
        'board': chess.Board(chess.STARTING_FEN),
        'engine': Stockfish(path=STOCKFISH_PATH)
    }
    return jsonify({'session_id': session_id, 'fen': chess.STARTING_FEN})

@app.route('/set-board-corners', methods=['POST'])
def set_board_corners():
    data = request.json
    session_id = data['session_id']
    corners = data['corners']
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
    session = sessions[session_id]
    corners = session['corners']
    if corners is None:
        return jsonify({'error': 'Corners not set'}), 400
    warped = get_board_perspective(frame, corners)
    # For the first frame, capture baseline
    if session['baseline_stats'] is None:
        starting_board = fen_to_board_array(chess.STARTING_FEN.split()[0])
        baseline_stats = capture_baseline_squares(warped, starting_board)
        session['baseline_stats'] = baseline_stats
    else:
        baseline_stats = session['baseline_stats']
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
    prev_fen = session['fen']
    session['prev_fen'] = prev_fen
    session['fen'] = fen
    detected_move = None
    engine_move = None
    # Only try to infer move if FEN changed
    if fen.split(' ')[0] != prev_fen.split(' ')[0]:
        board = session['board']
        # Try all legal moves to match new FEN
        move_found = False
        for m in board.legal_moves:
            temp_board = board.copy()
            temp_board.push(m)
            if temp_board.board_fen() == chess.Board(fen).board_fen():
                detected_move = m.uci()
                board.push(m)
                session['board'] = board
                session['fen'] = board.fen()
                move_found = True
                break
        if move_found:
            # If it's now white's turn (robot), do nothing (frontend will call /engine-move)
            # If it's now black's turn (human), just return
            return jsonify({'fen': session['fen'], 'detected_move': detected_move, 'turn': 'w' if board.turn == chess.WHITE else 'b'})
        else:
            return jsonify({'fen': fen, 'error': 'Could not infer move', 'turn': 'w' if session['board'].turn == chess.WHITE else 'b'})
    else:
        # No move detected
        return jsonify({'fen': fen, 'info': 'No move detected', 'turn': 'w' if session['board'].turn == chess.WHITE else 'b'})

@app.route('/engine-move', methods=['POST'])
def engine_move():
    data = request.json
    session_id = data['session_id']
    session = sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    board = session['board']
    if board.turn != chess.WHITE:
        return jsonify({'error': 'Not white\'s turn'}), 400
    session['engine'].set_fen_position(board.fen())
    engine_move = session['engine'].get_best_move_time(100)
    if engine_move:
        board.push(chess.Move.from_uci(engine_move))
        session['fen'] = board.fen()
        return jsonify({'fen': session['fen'], 'engine_move': engine_move})
    else:
        return jsonify({'error': 'No move found'}), 400

@app.route('/get-board-state', methods=['GET'])
def get_board_state():
    session_id = request.args.get('session_id')
    fen = sessions.get(session_id, {}).get('fen')
    if fen is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'fen': fen})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
