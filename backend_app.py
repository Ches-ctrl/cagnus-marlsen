from flask import Flask, Response, jsonify, request
import cv2
import threading
import time

from final_fen_live import UltimateChessDetector

app = Flask(__name__)

# No CORS enabled

detector = UltimateChessDetector(enable_visualization=False)
detector.initialize_camera(0)
detector.calibrated = False

frame_lock = threading.Lock()
latest_frame = None

def video_processing_loop():
    global latest_frame
    while True:
        ret, frame = detector.cap.read()
        if not ret:
            continue
        # Draw overlays (board, moves, etc.)
        if detector.calibrated:
            detector._draw_camera_overlay(frame, 0)
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.03)  # ~30 FPS

threading.Thread(target=video_processing_loop, daemon=True).start()

def generate_mjpeg():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fen')
def get_fen():
    return jsonify({
        'fen': detector.current_fen,
        'move_history': list(detector.move_history),
        'last_move': detector.last_move_description
    })

@app.route('/control', methods=['POST'])
def control():
    data = request.json
    action = data.get('action')
    if action == 'reset':
        detector._reset_position()
        return jsonify({'status': 'reset'})
    elif action == 'calibrate':
        detector.calibrate()
        return jsonify({'status': 'calibrated'})
    return jsonify({'status': 'unknown action'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
