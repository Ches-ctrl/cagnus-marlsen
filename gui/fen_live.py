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
        """Detect chess board corners using multiple methods."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Enhanced adaptive threshold
        corners = self._detect_corners_adaptive(gray)
        if corners is not None:
            return corners
        
        # Method 2: Canny edge detection
        corners = self._detect_corners_canny(gray)
        if corners is not None:
            return corners
        
        # Method 3: Different blur and threshold combinations
        corners = self._detect_corners_multi_threshold(gray)
        if corners is not None:
            return corners
        
        return None
    
    def _detect_corners_adaptive(self, gray):
        """Original adaptive threshold method with improvements."""
        # Try multiple blur levels
        for blur_size in [(5, 5), (7, 7), (9, 9), (3, 3)]:
            blurred = cv2.GaussianBlur(gray, blur_size, 0)
            
            # Try different adaptive threshold parameters
            for block_size in [11, 15, 19, 7]:
                for c_value in [2, 4, 6, 8]:
                    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, block_size, c_value)
                    
                    corners = self._find_rectangular_contour(thresh)
                    if corners is not None:
                        return corners
        return None
    
    def _detect_corners_canny(self, gray):
        """Canny edge detection method."""
        # Try different Canny parameters
        for low_thresh in [50, 100, 150]:
            for high_thresh in [150, 200, 250]:
                if high_thresh <= low_thresh:
                    continue
                
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, low_thresh, high_thresh)
                
                # Dilate edges to close gaps
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                corners = self._find_rectangular_contour(edges)
                if corners is not None:
                    return corners
        return None
    
    def _detect_corners_multi_threshold(self, gray):
        """Multiple threshold methods."""
        # Try Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners = self._find_rectangular_contour(thresh)
        if corners is not None:
            return corners
        
        # Try simple thresholding with different values
        for thresh_val in [100, 120, 140, 160, 180]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            corners = self._find_rectangular_contour(thresh)
            if corners is not None:
                return corners
        
        return None
    
    def _find_rectangular_contour(self, thresh):
        """Find the best rectangular contour from a thresholded image."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dynamic area limits based on image size
        image_area = thresh.shape[0] * thresh.shape[1]
        min_area = image_area * 0.05  # At least 5% of image
        max_area = image_area * 0.8   # At most 80% of image
        
        # Find rectangular contours
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Try different epsilon values for polygon approximation
                for epsilon_factor in [0.01, 0.02, 0.03, 0.015, 0.025]:
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        # Check if it's roughly rectangular
                        if self._is_roughly_rectangular(approx):
                            candidates.append((area, approx.reshape(4, 2)))
        
        # Return the largest valid candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return self.order_points(candidates[0][1])
        
        return None
    
    def _is_roughly_rectangular(self, approx):
        """Check if a 4-point contour is roughly rectangular."""
        points = approx.reshape(4, 2)
        
        # Calculate all 4 side lengths
        sides = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            sides.append(length)
        
        # Check if opposite sides are roughly equal (within 20% tolerance)
        sides.sort()
        if sides[0] == 0 or sides[2] == 0:
            return False
        
        ratio1 = sides[1] / sides[0]  # Short sides ratio
        ratio2 = sides[3] / sides[2]  # Long sides ratio
        
        # Both ratios should be close to 1, and short/long ratio should be reasonable
        if ratio1 > 1.2 or ratio2 > 1.2:
            return False
        
        # Check that it's not too skewed (aspect ratio check)
        aspect_ratio = sides[3] / sides[0]
        if aspect_ratio > 3 or aspect_ratio < 0.3:
            return False
        
        return True
    
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
        """Quick calibration process with enhanced feedback."""
        print("=== ENHANCED CALIBRATION ===")
        print("Position your board in starting position and press SPACE when ready...")
        print("Press 'd' to show detection debug info, 'm' for manual selection, 'q' to quit")
        
        show_debug = False
        detection_attempts = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Try automatic detection continuously for visual feedback
            corners = self.detect_board_corners(frame)
            detection_attempts += 1
            
            # Visual feedback
            if corners is not None:
                # Draw detected corners
                corners_int = corners.astype(int)
                cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)
                for i, corner in enumerate(corners_int):
                    cv2.circle(frame, tuple(corner), 8, (0, 255, 0), -1)
                    cv2.putText(frame, str(i+1), (corner[0]+10, corner[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "BOARD DETECTED! Press SPACE to confirm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Searching for board... (attempt {detection_attempts})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(frame, "SPACE: confirm | D: debug | M: manual | Q: quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if show_debug:
                # Show detection process
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                debug_frame = self._get_debug_visualization(gray)
                cv2.imshow('Debug - Detection Process', debug_frame)
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space pressed
                if corners is not None:
                    self.board_corners = corners
                    self.calibrated = True
                    
                    # Capture baseline immediately
                    board_image = self.get_board_perspective(frame, corners)
                    if board_image is not None:
                        self.capture_baseline(board_image)
                    
                    cv2.destroyAllWindows()
                    print("✓ Automatic detection successful!")
                    return True
                else:
                    print("No board detected. Try adjusting lighting or position.")
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow('Debug - Detection Process')
            elif key == ord('m'):
                cv2.destroyAllWindows()
                corners = self.manual_corner_selection()
                if corners is not None:
                    self.board_corners = corners
                    self.calibrated = True
                    
                    ret, frame = self.cap.read()
                    if ret:
                        board_image = self.get_board_perspective(frame, corners)
                        if board_image is not None:
                            self.capture_baseline(board_image)
                    print("✓ Manual selection completed!")
                    return True
                print("✗ Manual selection failed.")
                return False
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
    
    def _get_debug_visualization(self, gray):
        """Create a debug visualization showing the detection process."""
        # Create a grid of debug images
        h, w = gray.shape
        debug_h, debug_w = h // 3, w // 3
        
        # Resize gray image
        gray_small = cv2.resize(gray, (debug_w, debug_h))
        
        # Try adaptive threshold
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        thresh_adaptive_small = cv2.resize(thresh_adaptive, (debug_w, debug_h))
        
        # Try Canny
        edges = cv2.Canny(blurred, 100, 200)
        edges_small = cv2.resize(edges, (debug_w, debug_h))
        
        # Try Otsu
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_otsu_small = cv2.resize(thresh_otsu, (debug_w, debug_h))
        
        # Create debug grid
        debug_frame = np.zeros((debug_h * 2, debug_w * 2), dtype=np.uint8)
        debug_frame[0:debug_h, 0:debug_w] = gray_small
        debug_frame[0:debug_h, debug_w:debug_w*2] = thresh_adaptive_small
        debug_frame[debug_h:debug_h*2, 0:debug_w] = edges_small
        debug_frame[debug_h:debug_h*2, debug_w:debug_w*2] = thresh_otsu_small
        
        # Convert to color for labels
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(debug_frame, "Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_frame, "Adaptive", (debug_w + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_frame, "Canny", (5, debug_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_frame, "Otsu", (debug_w + 5, debug_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_frame
    
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