import cv2
import numpy as np
import time
from collections import Counter

class MoveTrackingFENDetector:
    """
    Webcam FEN detector focused on detecting moves and outputting FEN states.
    """

    def __init__(self, white_on_bottom=True):
        self.cap = None
        self.board_corners = None
        self.calibrated = False
        self.baseline_captured = False
        self.baseline_squares = None
        self.white_on_bottom = white_on_bottom

        # Starting position FEN - depends on board orientation
        if white_on_bottom:
            self.starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        else:
            self.starting_fen = "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w KQkq - 0 1"

        self.starting_board = self.fen_to_board_array(self.starting_fen.split()[0])
        self.current_fen = self.starting_fen
        self.last_stable_fen = self.starting_fen

        # Move detection parameters
        self.stability_frames = 10  # Frames position must be stable before reporting
        self.stable_count = 0
        self.min_contour_area = 5000
        self.max_contour_area = 200000

        print(f"=== MOVE TRACKING FEN DETECTOR ===")
        print(f"Board orientation: {'White on bottom' if white_on_bottom else 'Black on bottom'}")
        print(f"Starting FEN: {self.starting_fen}")
        print("")

    def initialize_camera(self, camera_index=0):
        """Initialize the webcam."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Let camera adjust
        for _ in range(10):
            self.cap.read()

    def fen_to_board_array(self, fen_board):
        """Convert FEN board string to 8x8 array."""
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

    def board_array_to_fen(self, board_array):
        """Convert 8x8 board array to FEN notation."""
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
        full_fen = f"{board_fen} w - - 0 1"

        return full_fen

    def detect_board_corners(self, frame):
        """Detect chess board corners using contour detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Try adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    return self.order_points(approx.reshape(4, 2))

        return None

    def order_points(self, pts):
        """Order points in top-left, top-right, bottom-right, bottom-left order."""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left

        return rect

    def manual_corner_selection(self):
        """Allow user to manually select board corners."""
        print("Manual corner selection - click corners in order: top-left, top-right, bottom-right, bottom-left")
        corners = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                print(f"Corner {len(corners)}: ({x}, {y})")

        cv2.namedWindow('Manual Selection')
        cv2.setMouseCallback('Manual Selection', mouse_callback)

        while len(corners) < 4:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Draw existing corners
            for i, corner in enumerate(corners):
                cv2.circle(frame, tuple(corner), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (corner[0]+10, corner[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Click corner {len(corners)+1}/4",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Manual Selection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('Manual Selection')
                return None

        cv2.destroyWindow('Manual Selection')
        return self.order_points(np.array(corners, dtype="float32"))

    def get_board_perspective(self, frame, corners):
        """Extract and warp the board to a square perspective."""
        if corners is None:
            return None

        dst = np.array([[0, 0], [640, 0], [640, 640], [0, 640]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(frame, matrix, (640, 640))

        return warped

    def extract_squares(self, board_image):
        """Extract individual squares from the warped board image."""
        squares = []
        square_size = 80  # 640 / 8

        for row in range(8):
            square_row = []
            for col in range(8):
                y1 = row * square_size
                y2 = (row + 1) * square_size
                x1 = col * square_size
                x2 = (col + 1) * square_size

                square = board_image[y1:y2, x1:x2]
                square_row.append(square)
            squares.append(square_row)

        return squares

    def capture_baseline(self, board_image):
        """Capture baseline images of each square in starting position."""
        print("Capturing baseline from starting position...")
        squares = self.extract_squares(board_image)

        self.baseline_squares = {}

        for row in range(8):
            for col in range(8):
                square_key = f"{row}_{col}"
                piece = self.starting_board[row][col]

                # Calculate square statistics
                gray = cv2.cvtColor(squares[row][col], cv2.COLOR_BGR2GRAY)

                self.baseline_squares[square_key] = {
                    'piece': piece,
                    'avg_brightness': np.mean(gray),
                    'std_brightness': np.std(gray),
                    'avg_color': np.mean(squares[row][col])
                }

        self.baseline_captured = True
        print("Baseline captured successfully!")

    def detect_piece_simple(self, square_image, row, col):
        """Simple piece detection using baseline comparison."""
        square_key = f"{row}_{col}"

        if not self.baseline_captured or square_key not in self.baseline_squares:
            return '.'

        baseline = self.baseline_squares[square_key]
        expected_piece = baseline['piece']

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
        current_avg = np.mean(gray)
        current_std = np.std(gray)

        # Calculate difference from baseline
        brightness_diff = abs(current_avg - baseline['avg_brightness'])
        std_diff = abs(current_std - baseline['std_brightness'])

        # If very similar to baseline, assume piece hasn't moved
        if brightness_diff < 15 and std_diff < 10:
            return expected_piece

        # If significantly different, check if square became empty or occupied
        if current_std < 15:  # Low variation suggests empty square
            return '.'

        # Try to determine piece color and type
        is_white_piece = current_avg > 120

        # Simple classification based on edge density
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

    def get_current_fen(self, frame):
        """Get current FEN from frame."""
        board_image = self.get_board_perspective(frame, self.board_corners)
        if board_image is None:
            return None

        squares = self.extract_squares(board_image)

        # Detect pieces in each square
        board_state = []
        for row in range(8):
            board_row = []
            for col in range(8):
                piece = self.detect_piece_simple(squares[row][col], row, col)
                board_row.append(piece)
            board_state.append(board_row)

        return self.board_array_to_fen(np.array(board_state))

    def calibrate_quick(self):
        """Quick calibration process."""
        print("=== QUICK CALIBRATION ===")
        print("Position your board in starting position and press SPACE when ready...")
        print("Or press 'm' for manual corner selection, 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            cv2.putText(frame, "Press SPACE when board is ready",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'm' for manual selection",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space pressed
                corners = self.detect_board_corners(frame)
                if corners is not None:
                    self.board_corners = corners
                    self.calibrated = True

                    # Capture baseline immediately
                    board_image = self.get_board_perspective(frame, corners)
                    if board_image is not None:
                        self.capture_baseline(board_image)

                    cv2.destroyWindow('Calibration')
                    return True
                else:
                    print("Could not detect board automatically. Try manual selection.")
            elif key == ord('m'):
                cv2.destroyWindow('Calibration')
                corners = self.manual_corner_selection()
                if corners is not None:
                    self.board_corners = corners
                    self.calibrated = True

                    ret, frame = self.cap.read()
                    if ret:
                        board_image = self.get_board_perspective(frame, corners)
                        if board_image is not None:
                            self.capture_baseline(board_image)
                    return True
                return False
            elif key == ord('q'):
                cv2.destroyWindow('Calibration')
                return False

    def run_move_tracking(self):
        """Main function for tracking moves and outputting FEN."""
        if not self.calibrate_quick():
            print("Calibration failed. Exiting.")
            return

        print("=== MOVE TRACKING ACTIVE ===")
        print(f"Initial FEN: {self.starting_fen}")
        print("Watching for moves... Press 'q' to quit, 'r' to reset to starting position")
        print("")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Get current FEN
            current_fen = self.get_current_fen(frame)
            if current_fen is None:
                continue

            # Check if position has changed
            if current_fen != self.current_fen:
                self.current_fen = current_fen
                self.stable_count = 0  # Reset stability counter
            else:
                self.stable_count += 1

            # Only report FEN if position has been stable
            if self.stable_count >= self.stability_frames and current_fen != self.last_stable_fen:
                print(f"NEW POSITION: {current_fen}")
                self.last_stable_fen = current_fen

            # Visual feedback
            corners = self.board_corners.astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

            status = "STABLE" if self.stable_count >= self.stability_frames else f"DETECTING... {self.stable_count}/{self.stability_frames}"
            cv2.putText(frame, f"Status: {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Move Tracking', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print(f"RESET TO: {self.starting_fen}")
                self.current_fen = self.starting_fen
                self.last_stable_fen = self.starting_fen
                self.stable_count = 0

    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function."""
    print("=== CHESS BOARD ORIENTATION ===")
    print("1. White pieces on bottom (standard)")
    print("2. Black pieces on bottom (flipped)")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            white_on_bottom = True
            break
        elif choice == '2':
            white_on_bottom = False
            break
        else:
            print("Please enter 1 or 2")

    detector = MoveTrackingFENDetector(white_on_bottom=white_on_bottom)

    try:
        detector.initialize_camera()
        detector.run_move_tracking()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
