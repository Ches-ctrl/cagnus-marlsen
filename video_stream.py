from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import threading
from final_fen_live import UltimateChessDetector
import time

app = Flask(__name__)
CORS(app)

# Camera index (change if needed)
CAMERA_INDEX = 0

# Store corners globally for now
corners = []

# Instantiate the chess detector
chess_detector = UltimateChessDetector(enable_visualization=False)

# Thread control
detection_thread = None
thread_lock = threading.Lock()

# Shared state for API
game_state = {
    'fen': None,
    'last_move': None,
    'move_history': [],
    'board_array': None,
}
game_state_lock = threading.Lock()

def gen_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_corners', methods=['POST'])
def set_corners():
    global corners, detection_thread
    data = request.get_json()
    corners = data.get('corners', [])
    print('Received corners:', corners)
    chess_detector.set_board_corners(corners)

    def detection_loop():
        try:
            chess_detector.initialize_camera(CAMERA_INDEX)
            while True:
                if not chess_detector.calibrated or chess_detector.board_corners is None:
                    time.sleep(0.1)
                    continue
                ret, frame = chess_detector.cap.read()
                if not ret:
                    continue
                fen = chess_detector.get_current_fen(frame)
                if fen:
                    with game_state_lock:
                        game_state['fen'] = fen
                        game_state['last_move'] = chess_detector.last_move_description
                        game_state['move_history'] = list(chess_detector.move_history)
                        board_array = chess_detector.fen_to_board_array(fen.split()[0])
                        game_state['board_array'] = board_array.tolist()
                time.sleep(0.5)
        except Exception as e:
            print(f"Detection loop error: {e}")

    with thread_lock:
        if detection_thread is None or not detection_thread.is_alive():
            detection_thread = threading.Thread(target=detection_loop, daemon=True)
            detection_thread.start()
            print("Started chess detection thread.")
        else:
            print("Detection thread already running.")
    return {'status': 'ok'}

@app.route('/game_state')
def get_game_state():
    with game_state_lock:
        return jsonify({
            'fen': game_state['fen'],
            'last_move': game_state['last_move'],
            'move_history': game_state['move_history'],
            'board_array': game_state['board_array'],
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
