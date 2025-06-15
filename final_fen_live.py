#!/usr/bin/env python3
"""
Ultimate Chess Move Detector
Combines the best features from multiple detection approaches for maximum accuracy and reliability.

Features:
- Multi-method board detection (chessboard pattern, contours, corners)
- Enhanced piece detection with template matching and statistical analysis
- Consensus-based stability checking with move validation
- Cross-platform compatibility with optional PyGame visualization
- Comprehensive calibration with debug modes
- Template learning with save/load functionality
- Adaptive lighting compensation
- Robust error handling and recovery
"""

import cv2
import numpy as np
import time
import os
import pickle
import json
from collections import Counter, deque
from typing import Optional, Tuple, List, Dict, Any
import argparse

# Optional PyGame for visualization
try:
    import pygame
    PYGAME_AVAILABLE = True
    # Suppress pygame startup message
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
except ImportError:
    PYGAME_AVAILABLE = False
    print("PyGame not available - visualization disabled")

# Optional sklearn for advanced clustering
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class UltimateChessDetector:
    """
    Ultimate chess move detector combining multiple detection methods for maximum accuracy.
    """

    def __init__(self, enable_visualization: bool = False):
        # Core setup
        self.cap = None
        self.board_corners = None
        self.calibrated = False

        # FIXED COORDINATE SYSTEM - Always the same
        # X axis: A, B, C, D, E, F, G, H (left to right)
        # Y axis: 1, 2, 3, 4, 5, 6, 7, 8 (bottom to top)
        # Black pieces always on bottom (ranks 1-2)
        self.files = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # left to right
        self.ranks = ['8', '7', '6', '5', '4', '3', '2', '1']   # top to bottom (visual order)
        self.rank_numbers = [8, 7, 6, 5, 4, 3, 2, 1]          # for display

        # Starting position with BLACK ON BOTTOM
        # Rank 1 (bottom): Black pieces (rnbqkbnr)
        # Rank 2: Black pawns (pppppppp)
        # Rank 7: White pawns (PPPPPPPP)
        # Rank 8 (top): White pieces (RNBQKBNR)
        self.starting_fen = "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w KQkq - 0 1"

        self.starting_board = self.fen_to_board_array(self.starting_fen.split()[0])
        self.current_fen = self.starting_fen
        self.last_stable_fen = self.starting_fen

        # Detection parameters
        self.stability_frames = 20  # Increased for better stability
        self.consensus_threshold = 0.85  # 85% agreement required
        self.detection_history = deque(maxlen=30)

        # Enhanced detection data
        self.baseline_data = {}
        self.piece_templates = {}
        self.empty_templates = {}
        self.lighting_baseline = None
        self.baseline_captured = False

        # Visualization
        self.enable_visualization = enable_visualization and PYGAME_AVAILABLE
        self.pygame_initialized = False
        self.pygame_screen = None
        self.piece_images = None

        # Detection settings
        self.detection_methods = {
            'template_matching': True,
            'histogram_analysis': True,
            'statistical_comparison': True,
            'edge_analysis': True,
            'motion_detection': True
        }

        # Display settings
        self.show_square_recognition = True  # Show blue dots and square labels

        # Move tracking
        self.last_move_description = None
        self.move_history = deque(maxlen=10)  # Store last 10 moves

        print(f"=== ULTIMATE CHESS MOVE DETECTOR ===")
        print(f"FIXED COORDINATE SYSTEM:")
        print(f"  X axis: A-H (left to right)")
        print(f"  Y axis: 1-8 (bottom to top)")
        print(f"  Black pieces on bottom (ranks 1-2)")
        print(f"  White pieces on top (ranks 7-8)")
        print(f"Starting FEN: {self.starting_fen}")
        print(f"Visualization: {'Enabled' if self.enable_visualization else 'Disabled'}")
        print("")

    def initialize_camera(self, camera_index: int = 0) -> None:
        """Initialize camera with optimal settings and robust error handling."""
        print(f"Initializing camera {camera_index}...")

        # Try different camera backends
        backends = [
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
            (cv2.CAP_V4L2, "V4L2 (Linux)"),
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_GSTREAMER, "GStreamer")
        ]

        for backend, name in backends:
            try:
                print(f"  Trying {name} backend...")
                self.cap = cv2.VideoCapture(camera_index, backend)

                if not self.cap.isOpened():
                    print(f"    ✗ Could not open camera with {name}")
                    continue

                # Test if camera actually provides frames
                print(f"    Testing frame capture...")
                success_count = 0
                for attempt in range(10):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        success_count += 1
                        print(f"    Frame {attempt + 1}: {frame.shape if frame is not None else 'None'}")
                    else:
                        print(f"    Frame {attempt + 1}: Failed")

                    if success_count >= 3:  # Need at least 3 successful frames
                        break

                    time.sleep(0.1)  # Small delay between attempts

                if success_count >= 3:
                    print(f"    ✓ {name} backend working!")
                    break
                else:
                    print(f"    ✗ {name} backend not providing consistent frames")
                    self.cap.release()
                    self.cap = None

            except Exception as e:
                print(f"    ✗ {name} backend failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                continue

        if self.cap is None or not self.cap.isOpened():
            self._show_camera_troubleshooting()
            raise Exception(f"Could not initialize any camera on index {camera_index}")

        # Set optimal camera properties (with error handling)
        print("Setting camera properties...")
        settings = [
            (cv2.CAP_PROP_FRAME_WIDTH, 1280, "Frame Width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, 720, "Frame Height"),
            (cv2.CAP_PROP_FPS, 30, "FPS"),
            (cv2.CAP_PROP_AUTOFOCUS, 1, "Autofocus"),
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "Auto Exposure")
        ]

        for prop, value, name in settings:
            try:
                old_value = self.cap.get(prop)
                self.cap.set(prop, value)
                new_value = self.cap.get(prop)
                print(f"  {name}: {old_value} → {new_value} (requested: {value})")
            except Exception as e:
                print(f"  {name}: Failed to set ({e})")

        # Extended camera warm-up with progress
        print("Camera warm-up...")
        successful_frames = 0
        for i in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                successful_frames += 1
                if i % 5 == 0:  # Print every 5th frame
                    print(f"  Frame {i+1}/30: ✓ {frame.shape}")
            else:
                print(f"  Frame {i+1}/30: ✗ Failed")

            time.sleep(0.1)

        if successful_frames < 20:
            raise Exception(f"Camera unstable: only {successful_frames}/30 frames successful")

        print(f"✓ Camera initialized successfully! ({successful_frames}/30 frames captured)")

        # Display current camera info
        self._show_camera_info()

    def _show_camera_troubleshooting(self) -> None:
        """Show camera troubleshooting information."""
        print("\n" + "="*60)
        print("CAMERA TROUBLESHOOTING")
        print("="*60)
        print("Camera initialization failed. Try these solutions:")
        print("")
        print("1. Check camera permissions:")
        print("   - macOS: System Preferences → Security & Privacy → Camera")
        print("   - Windows: Settings → Privacy → Camera")
        print("   - Linux: Check if user is in 'video' group")
        print("")
        print("2. Close other applications using the camera:")
        print("   - Zoom, Skype, FaceTime, Photo Booth, etc.")
        print("")
        print("3. Try different camera indices:")
        print("   python ultimate_chess_detector.py --camera 1")
        print("   python ultimate_chess_detector.py --camera 2")
        print("")
        print("4. List available cameras:")
        self._list_available_cameras()
        print("")
        print("5. Check camera physically:")
        print("   - Ensure USB camera is connected")
        print("   - Try a different USB port")
        print("   - Check camera LED (should be on)")
        print("="*60)

    def _list_available_cameras(self) -> None:
        """List all available cameras."""
        print("   Available cameras:")
        found_cameras = []

        for i in range(10):  # Check first 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        found_cameras.append(f"     Camera {i}: Available ({frame.shape})")
                    else:
                        found_cameras.append(f"     Camera {i}: Opens but no frames")
                    cap.release()
                else:
                    # Don't list unopenable cameras to reduce noise
                    pass
            except Exception as e:
                found_cameras.append(f"     Camera {i}: Error - {e}")

        if found_cameras:
            for camera in found_cameras:
                print(camera)
        else:
            print("     No cameras detected")

    def _show_camera_info(self) -> None:
        """Display current camera information."""
        if not self.cap or not self.cap.isOpened():
            return

        print("\nCamera Information:")
        properties = [
            (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
            (cv2.CAP_PROP_FPS, "FPS"),
            (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
            (cv2.CAP_PROP_CONTRAST, "Contrast"),
            (cv2.CAP_PROP_SATURATION, "Saturation"),
            (cv2.CAP_PROP_AUTOFOCUS, "Autofocus"),
            (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto Exposure")
        ]

        for prop, name in properties:
            try:
                value = self.cap.get(prop)
                print(f"  {name}: {value}")
            except:
                print(f"  {name}: Not available")

    def initialize_visualization(self) -> bool:
        """Initialize PyGame visualization if available."""
        if not self.enable_visualization or not PYGAME_AVAILABLE:
            return False

        try:
            pygame.init()

            # Board display settings
            self.square_size = 60
            self.board_size = self.square_size * 8
            self.screen_width = self.board_size + 300
            self.screen_height = self.board_size + 100

            self.pygame_screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Ultimate Chess Detector')

            # Load piece images if available
            self.piece_images = self._load_piece_images()

            # Initialize fonts
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

            self.pygame_initialized = True
            print("✓ PyGame visualization initialized")
            return True

        except Exception as e:
            print(f"PyGame initialization failed: {e}")
            self.enable_visualization = False
            return False

    def _load_piece_images(self) -> Optional[Dict[str, Any]]:
        """Load chess piece images for visualization."""
        piece_files = {
            'wp': 'white_pawn.png', 'bp': 'black_pawn.png',
            'wn': 'white_knight.png', 'bn': 'black_knight.png',
            'wb': 'white_bishop.png', 'bb': 'black_bishop.png',
            'wr': 'white_rook.png', 'br': 'black_rook.png',
            'wq': 'white_queen.png', 'bq': 'black_queen.png',
            'wk': 'white_king.png', 'bk': 'black_king.png'
        }

        # Try different image directories
        for img_dir in ['images', 'gui/images', 'assets', 'pieces']:
            if os.path.exists(img_dir):
                try:
                    pieces = {}
                    for piece_key, filename in piece_files.items():
                        # Try different extensions
                        for ext in ['.png', '.gif', '.jpg', '.jpeg']:
                            filepath = os.path.join(img_dir, filename.replace('.png', ext))
                            if os.path.exists(filepath):
                                pieces[piece_key] = pygame.image.load(filepath)
                                pieces[piece_key] = pygame.transform.scale(pieces[piece_key],
                                                                         (self.square_size, self.square_size))
                                break

                    if len(pieces) == 12:  # All pieces loaded
                        print(f"✓ Loaded piece images from {img_dir}")
                        return pieces

                except Exception as e:
                    print(f"Error loading images from {img_dir}: {e}")
                    continue

        print("No piece images found - using text representation")
        return None

    # ===== BOARD DETECTION METHODS =====

    def detect_board_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Comprehensive board detection using multiple methods with fallbacks.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Chessboard pattern detection (most accurate for standard boards)
        corners = self._detect_chessboard_pattern(gray)
        if corners is not None:
            return corners

        # Method 2: Enhanced contour detection
        corners = self._detect_enhanced_contours(gray)
        if corners is not None:
            return corners

        # Method 3: Harris corner detection
        corners = self._detect_harris_corners(gray)
        if corners is not None:
            return corners

        # Method 4: Line intersection method
        corners = self._detect_line_intersections(gray)
        if corners is not None:
            return corners

        return None

    def _detect_chessboard_pattern(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect using OpenCV's chessboard pattern detector."""
        for size in [(7, 7), (6, 6), (5, 5), (4, 4)]:
            ret, corners = cv2.findChessboardCorners(
                gray, size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Extract board corners from internal corners
                board_corners = self._extract_board_corners(corners.reshape(-1, 2), size)
                if board_corners is not None:
                    return self.order_points(board_corners)

        return None

    def _extract_board_corners(self, internal_corners: np.ndarray, grid_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract board corners from internal chessboard corners."""
        rows, cols = grid_size

        if len(internal_corners) != rows * cols:
            return None

        # Reshape to grid
        corner_grid = internal_corners.reshape(rows, cols, 2)

        # Calculate square dimensions
        square_width = np.linalg.norm(corner_grid[0, -1] - corner_grid[0, 0]) / (cols - 1)
        square_height = np.linalg.norm(corner_grid[-1, 0] - corner_grid[0, 0]) / (rows - 1)

        # Extrapolate to board corners
        tl = corner_grid[0, 0] - np.array([square_width, square_height])
        tr = corner_grid[0, -1] + np.array([square_width, -square_height])
        br = corner_grid[-1, -1] + np.array([square_width, square_height])
        bl = corner_grid[-1, 0] + np.array([-square_width, square_height])

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _detect_enhanced_contours(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced contour-based detection with multiple preprocessing methods."""
        for blur_kernel in [(5, 5), (7, 7), (9, 9)]:
            blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

            # Try different thresholding methods
            thresh_methods = [
                ('adaptive', lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                ('otsu', lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ('mean', lambda img: cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)[1])
            ]

            for method_name, thresh_func in thresh_methods:
                try:
                    thresh = thresh_func(blurred)
                    corners = self._find_best_rectangular_contour(thresh)
                    if corners is not None:
                        return corners
                except Exception:
                    continue

        return None

    def _detect_harris_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Harris corner detection with geometric constraints."""
        # Harris corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)

        # Find corner points
        corner_points = np.argwhere(corners > 0.01 * corners.max())
        corner_points = corner_points[:, [1, 0]]  # Convert to (x, y)

        if len(corner_points) < 4:
            return None

        # Find optimal rectangle
        return self._find_optimal_rectangle(corner_points)

    def _detect_line_intersections(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using line intersections."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is None or len(lines) < 4:
            return None

        # Group lines into horizontal and vertical
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            rho, theta = line[0]
            if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                horizontal_lines.append((rho, theta))
            elif abs(theta - np.pi/2) < np.pi/4:
                vertical_lines.append((rho, theta))

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None

        # Find intersections of extreme lines
        corners = []
        for h_line in [horizontal_lines[0], horizontal_lines[-1]]:
            for v_line in [vertical_lines[0], vertical_lines[-1]]:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    corners.append(intersection)

        if len(corners) == 4:
            return self.order_points(np.array(corners, dtype=np.float32))

        return None

    def _line_intersection(self, line1: Tuple[float, float], line2: Tuple[float, float]) -> Optional[np.ndarray]:
        """Find intersection of two lines in polar form."""
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]])
        b = np.array([rho1, rho2])

        try:
            intersection = np.linalg.solve(A, b)
            return intersection
        except np.linalg.LinAlgError:
            return None

    def _find_best_rectangular_contour(self, thresh: np.ndarray) -> Optional[np.ndarray]:
        """Find the best rectangular contour from thresholded image."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = thresh.shape[0] * thresh.shape[1]
        min_area = image_area * 0.05  # Minimum 5% of image
        max_area = image_area * 0.85  # Maximum 85% of image

        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Try different epsilon values
                for epsilon_factor in [0.01, 0.015, 0.02, 0.025, 0.03]:
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 4:
                        # Calculate rectangularity score
                        rectangularity = self._calculate_rectangularity(approx.reshape(4, 2))
                        score = area * rectangularity

                        if score > best_score:
                            best_score = score
                            best_contour = approx.reshape(4, 2)

        return self.order_points(best_contour) if best_contour is not None else None

    def _find_optimal_rectangle(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Find optimal rectangle from corner points."""
        if len(points) < 4:
            return None

        best_rect = None
        best_area = 0

        # Use clustering to group nearby points if we have many
        if len(points) > 8 and SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(points)
            # Use cluster centers as corner candidates
            corner_candidates = kmeans.cluster_centers_
        else:
            corner_candidates = points

        # Try combinations of 4 points
        from itertools import combinations
        for combo in combinations(corner_candidates, 4):
            rect_points = np.array(combo, dtype=np.float32)

            if self._is_valid_rectangle(rect_points):
                area = cv2.contourArea(rect_points)
                if area > best_area:
                    best_area = area
                    best_rect = rect_points

        return self.order_points(best_rect) if best_rect is not None else None

    def _calculate_rectangularity(self, points: np.ndarray) -> float:
        """Calculate how rectangular a 4-point shape is (0-1 score)."""
        if len(points) != 4:
            return 0

        # Calculate angles at each corner
        angles = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return 0

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

        # Perfect rectangle has all 90-degree angles
        target_angle = np.pi / 2
        angle_deviations = [abs(angle - target_angle) for angle in angles]
        max_deviation = max(angle_deviations)

        # Return score (1 = perfect rectangle, 0 = very poor)
        return max(0, 1 - (max_deviation / (np.pi / 4)))

    def _is_valid_rectangle(self, points: np.ndarray) -> bool:
        """Check if 4 points form a valid rectangle."""
        if len(points) != 4:
            return False

        # Calculate all pairwise distances
        distances = []
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.linalg.norm(points[i] - points[j])
                distances.append(d)

        distances.sort()

        # Check aspect ratio (not too elongated)
        sides = distances[:4]
        diagonals = distances[4:]

        if len(sides) < 4 or len(diagonals) < 2:
            return False

        # Allow reasonable tolerance
        tolerance = 0.15

        # Check diagonals are approximately equal
        if abs(diagonals[0] - diagonals[1]) > tolerance * max(diagonals):
            return False

        # Check we have two pairs of equal sides
        sides.sort()
        if (abs(sides[0] - sides[1]) > tolerance * max(sides) or
            abs(sides[2] - sides[3]) > tolerance * max(sides)):
            return False

        # Check aspect ratio
        min_side = min(sides[:2])
        max_side = max(sides[2:])
        if min_side > 0 and (max_side / min_side) > 4:  # Not too elongated
            return False

        return True

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order."""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left

        return rect

    # ===== BOARD PROCESSING =====

    def get_board_perspective(self, frame: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """Extract and warp the board to a square perspective."""
        if corners is None:
            return None

        # Use higher resolution for better piece detection
        dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(frame, matrix, (800, 800))

        return warped

    def extract_squares(self, board_image: np.ndarray) -> List[List[np.ndarray]]:
        """Extract individual squares with proper padding."""
        squares = []
        square_size = 100  # 800 / 8
        padding = 8  # Padding to avoid edge effects

        for row in range(8):
            square_row = []
            for col in range(8):
                y1 = row * square_size + padding
                y2 = (row + 1) * square_size - padding
                x1 = col * square_size + padding
                x2 = (col + 1) * square_size - padding

                square = board_image[y1:y2, x1:x2]
                square_row.append(square)
            squares.append(square_row)

        return squares

    # ===== BASELINE AND TEMPLATE MANAGEMENT =====

    def print_board_state(self, fen: str = None, title: str = "BOARD STATE") -> None:
        """
        Print a detailed view of the current board state showing all squares and their contents.
        """
        if fen is None:
            fen = self.current_fen

        board_array = self.fen_to_board_array(fen.split()[0])

        print(f"\n=== {title} ===")

        # Print in a grid format that matches the visual board
        print("     A        B        C        D        E        F        G        H")
        print("  " + "─" * 73)

        for row in range(8):
            rank = self.rank_numbers[row]
            line = f"{rank} │"

            for col in range(8):
                piece = board_array[row][col]
                square_name = f"{self.files[col]}{rank}"

                if piece == '.':
                    content = "  Empty  "
                else:
                    piece_name = self._get_short_piece_name(piece)
                    # Truncate long names to fit in column
                    if len(piece_name) > 8:
                        content = piece_name[:8]
                    else:
                        content = f"{piece_name:^8}"

                line += f" {content} │"

            print(line)
            if row < 7:  # Don't print separator after last row
                print("  " + "─" * 73)

        print("  " + "─" * 73)

        # Print summary statistics
        piece_counts = self._count_pieces(board_array)
        print(f"\nPiece Summary:")
        print(f"White pieces: {piece_counts['white']} | Black pieces: {piece_counts['black']} | Empty squares: {piece_counts['empty']}")

        # Print detailed piece inventory
        white_pieces = []
        black_pieces = []

        for row in range(8):
            for col in range(8):
                piece = board_array[row][col]
                square_name = f"{self.files[col]}{self.rank_numbers[row]}"

                if piece != '.':
                    piece_name = self._get_piece_name(piece)
                    piece_info = f"{piece_name} on {square_name}"

                    if piece.isupper():
                        white_pieces.append(piece_info)
                    else:
                        black_pieces.append(piece_info)

        if white_pieces:
            print(f"\nWhite pieces ({len(white_pieces)}):")
            for i, piece_info in enumerate(white_pieces):
                if i % 2 == 0:
                    print(f"  {piece_info:<35}", end="")
                else:
                    print(f" {piece_info}")
            if len(white_pieces) % 2 == 1:
                print()  # Add newline if odd number

        if black_pieces:
            print(f"\nBlack pieces ({len(black_pieces)}):")
            for i, piece_info in enumerate(black_pieces):
                if i % 2 == 0:
                    print(f"  {piece_info:<35}", end="")
                else:
                    print(f" {piece_info}")
            if len(black_pieces) % 2 == 1:
                print()  # Add newline if odd number

        print("=" * len(f"=== {title} ==="))

    def _count_pieces(self, board_array: np.ndarray) -> Dict[str, int]:
        """Count pieces on the board."""
        white_count = 0
        black_count = 0
        empty_count = 0

        for row in range(8):
            for col in range(8):
                piece = board_array[row][col]
                if piece == '.':
                    empty_count += 1
                elif piece.isupper():
                    white_count += 1
                else:
                    black_count += 1

        return {
            'white': white_count,
            'black': black_count,
            'empty': empty_count
        }

    def print_square_by_square_analysis(self, fen: str = None) -> None:
        """Print a detailed square-by-square analysis."""
        if fen is None:
            fen = self.current_fen

        board_array = self.fen_to_board_array(fen.split()[0])

        print(f"\n=== SQUARE-BY-SQUARE ANALYSIS ===")

        for row in range(8):
            rank = self.rank_numbers[row]
            print(f"\nRank {rank}:")

            rank_info = []
            for col in range(8):
                piece = board_array[row][col]
                square_name = f"{self.files[col]}{rank}"

                if piece == '.':
                    content = f"{square_name}: Empty"
                else:
                    piece_name = self._get_piece_name(piece)
                    content = f"{square_name}: {piece_name}"

                rank_info.append(content)

            # Print 4 squares per line for readability
            for i in range(0, len(rank_info), 4):
                line_squares = rank_info[i:i+4]
                print("  " + " | ".join(f"{square:<20}" for square in line_squares))

        print("=" * 35)

    def capture_comprehensive_baseline(self, board_image: np.ndarray) -> None:
        """Capture comprehensive baseline data for all squares."""
        print("Capturing comprehensive baseline data...")

        squares = self.extract_squares(board_image)

        # Initialize storage
        self.baseline_data = {}
        self.piece_templates = {}
        self.empty_templates = {'light': [], 'dark': []}

        # Global lighting reference
        gray_board = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        self.lighting_baseline = np.mean(gray_board)

        for row in range(8):
            for col in range(8):
                square_key = f"{row}_{col}"
                piece = self.starting_board[row][col]
                square_img = squares[row][col]

                # Comprehensive feature extraction
                features = self._extract_square_features(square_img)
                features['expected_piece'] = piece

                self.baseline_data[square_key] = features

                # Store templates by piece type
                if piece != '.':
                    if piece not in self.piece_templates:
                        self.piece_templates[piece] = []
                    self.piece_templates[piece].append(features['template'])
                else:
                    # Store empty square template by color
                    square_color = 'light' if (row + col) % 2 == 0 else 'dark'
                    self.empty_templates[square_color].append(features['template'])

        self.baseline_captured = True

        print(f"✓ Baseline captured:")
        print(f"  - {len(self.baseline_data)} squares analyzed")
        print(f"  - {len(self.piece_templates)} piece types")
        print(f"  - {sum(len(templates) for templates in self.empty_templates.values())} empty squares")

        # Print detailed initial board state
        self.print_board_state(self.starting_fen, "INITIAL BOARD STATE DETECTED")

    def _extract_square_features(self, square_img: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features from a square image."""
        # Convert to different color spaces
        gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(square_img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(square_img, cv2.COLOR_BGR2LAB)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate features
        features = {
            'template': gray.copy(),

            # Color statistics
            'mean_bgr': np.mean(square_img, axis=(0, 1)),
            'std_bgr': np.std(square_img, axis=(0, 1)),
            'mean_gray': np.mean(gray),
            'std_gray': np.std(gray),
            'mean_hsv': np.mean(hsv, axis=(0, 1)),
            'mean_lab': np.mean(lab, axis=(0, 1)),

            # Histograms
            'hist_bgr': [cv2.calcHist([square_img], [i], None, [32], [0, 256]) for i in range(3)],
            'hist_hsv': [cv2.calcHist([hsv], [i], None, [32], [0, 256]) for i in range(3)],

            # Edge features
            'edges': edges,
            'edge_density': np.sum(edges > 0) / edges.size,

            # Texture features
            'variance': np.var(gray),
            'contrast': np.std(gray),

            # Geometric features
            'contours': cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        }

        return features

    # ===== PIECE DETECTION =====

    def detect_piece_ultimate(self, square_img: np.ndarray, row: int, col: int) -> str:
        """
        Ultimate piece detection using multiple methods and consensus.
        """
        square_key = f"{row}_{col}"

        if not self.baseline_captured or square_key not in self.baseline_data:
            return '.'

        baseline = self.baseline_data[square_key]
        expected_piece = baseline['expected_piece']

        # Extract current features
        current_features = self._extract_square_features(square_img)

        # Apply all detection methods
        scores = {}

        if self.detection_methods['template_matching']:
            scores['template'] = self._score_template_matching(current_features, baseline)

        if self.detection_methods['histogram_analysis']:
            scores['histogram'] = self._score_histogram_similarity(current_features, baseline)

        if self.detection_methods['statistical_comparison']:
            scores['statistical'] = self._score_statistical_similarity(current_features, baseline)

        if self.detection_methods['edge_analysis']:
            scores['edge'] = self._score_edge_similarity(current_features, baseline)

        if self.detection_methods['motion_detection']:
            scores['motion'] = self._score_motion_similarity(current_features, baseline)

        # Weighted combination of scores
        weights = {
            'template': 0.30,
            'histogram': 0.20,
            'statistical': 0.20,
            'edge': 0.15,
            'motion': 0.15
        }

        combined_score = sum(scores.get(method, 0) * weight
                           for method, weight in weights.items()
                           if method in scores)

        # Decision thresholds
        if combined_score > 0.85:  # Very high confidence: same piece
            return expected_piece
        elif combined_score > 0.70:  # High confidence: likely same piece
            return expected_piece
        elif combined_score < 0.40:  # Low confidence: significant change
            return self._classify_new_piece(current_features, row, col)
        else:  # Medium confidence: additional analysis needed
            return self._resolve_ambiguous_detection(current_features, baseline, row, col)

    def _score_template_matching(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Score template matching similarity."""
        current_template = current['template']
        baseline_template = baseline['template']

        if current_template.shape != baseline_template.shape:
            current_template = cv2.resize(current_template,
                                        (baseline_template.shape[1], baseline_template.shape[0]))

        # Normalize for lighting
        current_norm = cv2.equalizeHist(current_template)
        baseline_norm = cv2.equalizeHist(baseline_template)

        # Multiple template matching methods
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        scores = []

        for method in methods:
            result = cv2.matchTemplate(current_norm, baseline_norm, method)
            scores.append(result[0, 0])

        return np.mean(scores)

    def _score_histogram_similarity(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Score histogram similarity."""
        scores = []

        # Compare all histogram types
        for hist_type in ['hist_bgr', 'hist_hsv']:
            if hist_type in current and hist_type in baseline:
                for i in range(len(current[hist_type])):
                    score = cv2.compareHist(current[hist_type][i],
                                          baseline[hist_type][i],
                                          cv2.HISTCMP_CORREL)
                    scores.append(max(0, score))  # Ensure non-negative

        return np.mean(scores) if scores else 0

    def _score_statistical_similarity(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Score statistical feature similarity with lighting compensation."""
        # Lighting compensation
        current_lighting = current['mean_gray']
        baseline_lighting = baseline['mean_gray']
        lighting_ratio = baseline_lighting / current_lighting if current_lighting > 0 else 1

        # Compensate current values
        adjusted_mean_gray = current['mean_gray'] * lighting_ratio
        adjusted_mean_bgr = current['mean_bgr'] * lighting_ratio

        # Calculate similarity scores
        scores = []

        # Gray value similarity
        gray_diff = abs(adjusted_mean_gray - baseline['mean_gray'])
        gray_score = max(0, 1 - gray_diff / 100)  # Normalize to 0-1
        scores.append(gray_score)

        # Color similarity
        color_diff = np.mean(np.abs(adjusted_mean_bgr - baseline['mean_bgr']))
        color_score = max(0, 1 - color_diff / 50)
        scores.append(color_score)

        # Texture similarity
        contrast_diff = abs(current['contrast'] - baseline['contrast'])
        contrast_score = max(0, 1 - contrast_diff / 30)
        scores.append(contrast_score)

        return np.mean(scores)

    def _score_edge_similarity(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Score edge feature similarity."""
        # Edge density similarity
        density_diff = abs(current['edge_density'] - baseline['edge_density'])
        density_score = max(0, 1 - density_diff / 0.3)

        # Edge template matching
        current_edges = current['edges']
        baseline_edges = baseline['edges']

        if current_edges.shape == baseline_edges.shape:
            edge_match = cv2.matchTemplate(current_edges, baseline_edges, cv2.TM_CCOEFF_NORMED)[0, 0]
            edge_score = max(0, edge_match)
        else:
            edge_score = 0

        return (density_score + edge_score) / 2

    def _score_motion_similarity(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        """Score motion/change detection."""
        current_template = current['template']
        baseline_template = baseline['template']

        if current_template.shape != baseline_template.shape:
            current_template = cv2.resize(current_template,
                                        (baseline_template.shape[1], baseline_template.shape[0]))

        # Frame difference
        diff = cv2.absdiff(current_template, baseline_template)
        _, thresh_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate motion score (higher = more motion)
        motion_ratio = np.sum(thresh_diff > 0) / thresh_diff.size

        # Convert to similarity score (less motion = higher similarity)
        return max(0, 1 - motion_ratio * 2)

    def _classify_new_piece(self, features: Dict[str, Any], row: int, col: int) -> str:
        """Classify what piece is now on the square."""
        # First check if square is empty
        if self._is_square_empty(features, row, col):
            return '.'

        # Determine piece color
        is_white = self._determine_piece_color(features)

        # Template matching against known pieces
        best_piece = None
        best_score = 0

        current_template = features['template']

        for piece_type, templates in self.piece_templates.items():
            # Only consider pieces of the correct color
            piece_is_white = piece_type.isupper()
            if is_white != piece_is_white:
                continue

            for template in templates:
                if current_template.shape != template.shape:
                    resized_current = cv2.resize(current_template,
                                               (template.shape[1], template.shape[0]))
                else:
                    resized_current = current_template

                # Normalize both images
                norm_current = cv2.equalizeHist(resized_current)
                norm_template = cv2.equalizeHist(template)

                score = cv2.matchTemplate(norm_current, norm_template, cv2.TM_CCOEFF_NORMED)[0, 0]
                if score > best_score:
                    best_score = score
                    best_piece = piece_type

        # If template matching found a good match, return it
        if best_piece is not None and best_score > 0.6:
            return best_piece

        # Fallback to feature-based classification
        return self._classify_by_features(features, is_white)

    def _is_square_empty(self, features: Dict[str, Any], row: int, col: int) -> bool:
        """Determine if square is empty using multiple criteria."""
        square_color = 'light' if (row + col) % 2 == 0 else 'dark'

        # Compare with empty square templates
        if square_color in self.empty_templates and self.empty_templates[square_color]:
            best_score = 0
            current_template = features['template']

            for template in self.empty_templates[square_color]:
                if current_template.shape != template.shape:
                    resized_current = cv2.resize(current_template,
                                               (template.shape[1], template.shape[0]))
                else:
                    resized_current = current_template

                score = cv2.matchTemplate(resized_current, template, cv2.TM_CCOEFF_NORMED)[0, 0]
                best_score = max(best_score, score)

            if best_score > 0.8:
                return True

        # Fallback criteria
        edge_density = features['edge_density']
        variance = features['variance']

        return edge_density < 0.08 and variance < 300

    def _determine_piece_color(self, features: Dict[str, Any]) -> bool:
        """Determine if piece is white (True) or black (False)."""
        # Use multiple color space information
        lab_lightness = features['mean_lab'][0]  # L channel from LAB
        hsv_value = features['mean_hsv'][2]      # V channel from HSV
        gray_mean = features['mean_gray']        # Grayscale mean

        # Weighted decision
        lightness_score = lab_lightness / 255
        value_score = hsv_value / 255
        gray_score = gray_mean / 255

        combined_score = (lightness_score * 0.4 + value_score * 0.3 + gray_score * 0.3)

        return combined_score > 0.45  # Threshold for white vs black

    def _classify_by_features(self, features: Dict[str, Any], is_white: bool) -> str:
        """Classify piece type based on shape and edge features."""
        edge_density = features['edge_density']
        variance = features['variance']
        contours = features['contours']

        # Default to pawn if unclear
        default_piece = 'P' if is_white else 'p'

        if not contours:
            return default_piece

        # Analyze largest contour
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)

        if area < 100:  # Very small contour
            return default_piece

        perimeter = cv2.arcLength(main_contour, True)
        if perimeter == 0:
            return default_piece

        # Shape descriptors
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 1
        extent = float(area) / (w * h) if (w * h) > 0 else 0

        # Convex hull
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Classification based on features
        piece_type = 'P'  # Default to pawn

        if edge_density < 0.15 and variance < 400:
            piece_type = 'P'  # Simple shape - pawn
        elif circularity > 0.6 and solidity > 0.8:
            piece_type = 'P'  # Round, solid - pawn
        elif aspect_ratio > 1.5 or aspect_ratio < 0.7:
            if extent > 0.7:
                piece_type = 'R'  # Tall/wide and solid - rook
            else:
                piece_type = 'Q'  # Tall/wide but complex - queen
        elif solidity < 0.6 or edge_density > 0.4:
            piece_type = 'N'  # Complex shape - knight
        elif circularity > 0.4 and edge_density > 0.25:
            piece_type = 'B'  # Moderately complex - bishop
        elif edge_density > 0.35:
            piece_type = 'K'  # Very complex - king

        return piece_type if is_white else piece_type.lower()

    def _resolve_ambiguous_detection(self, features: Dict[str, Any], baseline: Dict[str, Any], row: int, col: int) -> str:
        """Resolve ambiguous detections using additional analysis."""
        expected_piece = baseline['expected_piece']

        # Use temporal consistency if available
        if len(self.detection_history) > 5:
            recent_fens = list(self.detection_history)[-5:]
            position_history = []

            for fen in recent_fens:
                board = self.fen_to_board_array(fen.split()[0])
                position_history.append(board[row][col])

            # If position was consistently the same recently, bias toward that
            position_counter = Counter(position_history)
            most_common = position_counter.most_common(1)[0]

            if most_common[1] >= 4:  # Consistent in 4 out of 5 recent frames
                return most_common[0]

        # If still ambiguous, be conservative and return expected piece
        return expected_piece

    # ===== FEN PROCESSING =====

    def fen_to_board_array(self, fen_board: str) -> np.ndarray:
        """Convert FEN board string to 8x8 numpy array."""
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

    def board_array_to_fen(self, board_array: np.ndarray) -> str:
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
        return f"{board_fen} w - - 0 1"

    def get_current_fen(self, frame: np.ndarray) -> Optional[str]:
        """Get current FEN from camera frame using ultimate detection."""
        if not self.calibrated or self.board_corners is None:
            return None

        # Get board perspective
        board_image = self.get_board_perspective(frame, self.board_corners)
        if board_image is None:
            return None

        # Capture baseline on first successful detection
        if not self.baseline_captured:
            self.capture_comprehensive_baseline(board_image)
            return self.starting_fen

        # Extract squares and detect pieces
        squares = self.extract_squares(board_image)
        board_state = []

        for row in range(8):
            board_row = []
            for col in range(8):
                piece = self.detect_piece_ultimate(squares[row][col], row, col)
                board_row.append(piece)
            board_state.append(board_row)

        return self.board_array_to_fen(np.array(board_state))

    def analyze_move(self, old_fen: str, new_fen: str) -> Optional[str]:
        """
        Analyze the difference between two board positions and determine the move made.
        Returns move in descriptive notation (e.g., "White Pawn e2 to e4", "Black Knight b8 to c6")
        """
        old_board = self.fen_to_board_array(old_fen.split()[0])
        new_board = self.fen_to_board_array(new_fen.split()[0])

        # Find all differences between boards
        differences = []
        for row in range(8):
            for col in range(8):
                if old_board[row][col] != new_board[row][col]:
                    square_name = f"{self.files[col]}{self.rank_numbers[row]}"
                    differences.append({
                        'square': square_name,
                        'row': row,
                        'col': col,
                        'old_piece': old_board[row][col],
                        'new_piece': new_board[row][col]
                    })

        if len(differences) == 0:
            return None

        # Analyze different types of moves
        if len(differences) == 2:
            return self._analyze_normal_move(differences)
        elif len(differences) == 3:
            return self._analyze_en_passant(differences)
        elif len(differences) == 4:
            return self._analyze_castling(differences)
        else:
            return self._analyze_complex_move(differences)

    def _analyze_normal_move(self, differences: List[Dict]) -> Optional[str]:
        """Analyze a normal move (piece goes from one square to another)."""
        # Find the square that became empty and the square that got a piece
        from_square = None
        to_square = None
        piece_moved = None
        captured_piece = None

        for diff in differences:
            if diff['old_piece'] != '.' and diff['new_piece'] == '.':
                # This square became empty - piece moved FROM here
                from_square = diff['square']
                piece_moved = diff['old_piece']
            elif diff['old_piece'] == '.' and diff['new_piece'] != '.':
                # This square got a piece - piece moved TO here
                to_square = diff['square']
            elif diff['old_piece'] != '.' and diff['new_piece'] != '.':
                # This square had a piece and now has a different piece - capture
                to_square = diff['square']
                captured_piece = diff['old_piece']
                if piece_moved is None:  # In case we haven't found the from square yet
                    piece_moved = diff['new_piece']

        if from_square and to_square and piece_moved:
            piece_name = self._get_piece_name(piece_moved)
            move_text = f"{piece_name} {from_square} to {to_square}"

            if captured_piece:
                captured_name = self._get_piece_name(captured_piece)
                move_text += f" (captures {captured_name})"

            return move_text

        return None

    def _analyze_en_passant(self, differences: List[Dict]) -> Optional[str]:
        """Analyze en passant move (3 squares change)."""
        pawn_from = None
        pawn_to = None
        captured_square = None

        for diff in differences:
            if diff['old_piece'].lower() == 'p' and diff['new_piece'] == '.':
                if abs(diff['row'] - differences[0]['row']) == 1:  # Adjacent row
                    captured_square = diff['square']
                else:
                    pawn_from = diff['square']
            elif diff['old_piece'] == '.' and diff['new_piece'].lower() == 'p':
                pawn_to = diff['square']

        if pawn_from and pawn_to and captured_square:
            piece_moved = None
            for diff in differences:
                if diff['square'] == pawn_from:
                    piece_moved = diff['old_piece']
                    break

            if piece_moved:
                piece_name = self._get_piece_name(piece_moved)
                return f"{piece_name} {pawn_from} to {pawn_to} (en passant, captures pawn on {captured_square})"

        return None

    def _analyze_castling(self, differences: List[Dict]) -> Optional[str]:
        """Analyze castling move (4 squares change)."""
        king_from = None
        king_to = None
        rook_from = None
        rook_to = None

        for diff in differences:
            if diff['old_piece'].lower() == 'k' and diff['new_piece'] == '.':
                king_from = diff['square']
            elif diff['old_piece'] == '.' and diff['new_piece'].lower() == 'k':
                king_to = diff['square']
            elif diff['old_piece'].lower() == 'r' and diff['new_piece'] == '.':
                rook_from = diff['square']
            elif diff['old_piece'] == '.' and diff['new_piece'].lower() == 'r':
                rook_to = diff['square']

        if king_from and king_to and rook_from and rook_to:
            # Determine if it's kingside or queenside castling
            king_from_col = ord(king_from[0]) - ord('A')
            king_to_col = ord(king_to[0]) - ord('A')

            if king_to_col > king_from_col:
                castle_type = "kingside"
            else:
                castle_type = "queenside"

            # Determine color
            for diff in differences:
                if diff['square'] == king_from:
                    king_piece = diff['old_piece']
                    color = "White" if king_piece.isupper() else "Black"
                    break

            return f"{color} castles {castle_type} (King {king_from} to {king_to}, Rook {rook_from} to {rook_to})"

        return None

    def _analyze_complex_move(self, differences: List[Dict]) -> Optional[str]:
        """Analyze complex moves or multiple piece changes."""
        if len(differences) > 6:
            return f"Complex position change ({len(differences)} squares changed)"

        # Try to find the most likely single move among the differences
        moves = []
        for i, diff1 in enumerate(differences):
            for j, diff2 in enumerate(differences):
                if i != j:
                    # Check if this could be a from->to move
                    if (diff1['old_piece'] != '.' and diff1['new_piece'] == '.' and
                        diff2['old_piece'] != diff2['new_piece'] and diff2['new_piece'] != '.'):

                        if diff1['old_piece'] == diff2['new_piece']:  # Same piece
                            piece_name = self._get_piece_name(diff1['old_piece'])
                            move_text = f"{piece_name} {diff1['square']} to {diff2['square']}"

                            if diff2['old_piece'] != '.':
                                captured_name = self._get_piece_name(diff2['old_piece'])
                                move_text += f" (captures {captured_name})"

                            moves.append(move_text)

        if moves:
            return moves[0]  # Return the first valid move found

        return f"Position change detected ({len(differences)} squares modified)"

    def _get_short_piece_name(self, piece: str) -> str:
        """Convert piece symbol to short descriptive name for grid display."""
        color_short = "W" if piece.isupper() else "B"

        piece_names = {
            'p': 'Pawn',
            'n': 'Knight',
            'b': 'Bishop',
            'r': 'Rook',
            'q': 'Queen',
            'k': 'King'
        }

        piece_type = piece_names.get(piece.lower(), 'Unknown')
        return f"{color_short}{piece_type}"

    def _get_piece_name(self, piece: str) -> str:
        """Convert piece symbol to descriptive name."""
        color = "White" if piece.isupper() else "Black"

        piece_names = {
            'p': 'Pawn',
            'n': 'Knight',
            'b': 'Bishop',
            'r': 'Rook',
            'q': 'Queen',
            'k': 'King'
        }

        piece_type = piece_names.get(piece.lower(), 'Unknown')
        return f"{color} {piece_type}"

    def validate_fen_transition(self, new_fen: str) -> bool:
        """Validate that a FEN transition represents a reasonable move."""
        if new_fen == self.last_stable_fen:
            return True

        old_board = self.fen_to_board_array(self.last_stable_fen.split()[0])
        new_board = self.fen_to_board_array(new_fen.split()[0])

        # Count differences
        differences = []
        for row in range(8):
            for col in range(8):
                if old_board[row][col] != new_board[row][col]:
                    differences.append((row, col, old_board[row][col], new_board[row][col]))

        # No differences = same position (valid)
        if len(differences) == 0:
            return True

        # Single square change is usually noise
        if len(differences) == 1:
            return False

        # Too many changes suggests detection error
        if len(differences) > 8:
            return False

        # Check piece count consistency
        old_pieces = sum(1 for row in old_board for piece in row if piece != '.')
        new_pieces = sum(1 for row in new_board for piece in row if piece != '.')

        # Piece count can only decrease by 0 or 1 (capture)
        if new_pieces > old_pieces or (old_pieces - new_pieces) > 1:
            return False

        return True

    # ===== CALIBRATION =====

    def print_expected_starting_position(self) -> None:
        """Print the FIXED starting position layout."""
        print(f"\n=== FIXED BOARD LAYOUT ===")
        print(f"Coordinate System: A-H (left to right), 1-8 (bottom to top)")
        print(f"Black pieces always on bottom")
        print("")

        print("Your board should ALWAYS look like this:")
        print("     A    B    C    D    E    F    G    H")
        print("  ┌────┬────┬────┬────┬────┬────┬────┬────┐")
        print("8 │ ♖  │ ♘  │ ♗  │ ♕  │ ♔  │ ♗  │ ♘  │ ♖  │ ← White pieces (far from you)")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("7 │ ♙  │ ♙  │ ♙  │ ♙  │ ♙  │ ♙  │ ♙  │ ♙  │ ← White pawns")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("6 │    │    │    │    │    │    │    │    │")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("5 │    │    │    │    │    │    │    │    │")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("4 │    │    │    │    │    │    │    │    │")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("3 │    │    │    │    │    │    │    │    │")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("2 │ ♟  │ ♟  │ ♟  │ ♟  │ ♟  │ ♟  │ ♟  │ ♟  │ ← Black pawns")
        print("  ├────┼────┼────┼────┼────┼────┼────┼────┤")
        print("1 │ ♜  │ ♞  │ ♝  │ ♛  │ ♚  │ ♝  │ ♞  │ ♜  │ ← Black pieces (closest to you)")
        print("  └────┴────┴────┴────┴────┴────┴────┴────┘")
        print("")
        print("Key points:")
        print("  • A1 = Bottom-left corner (Black rook)")
        print("  • H1 = Bottom-right corner (Black rook)")
        print("  • A8 = Top-left corner (White rook)")
        print("  • H8 = Top-right corner (White rook)")
        print("  • Black King on E1, White King on E8")
        print("=============================")


    def calibrate(self) -> bool:
        """Comprehensive calibration with fixed coordinate system."""
        self.print_expected_starting_position()

        print("\n=== ULTIMATE CALIBRATION ===")
        print("1. FIRST: Set up your board to match the layout above")
        print("2. Make sure black pieces are on the bottom (closest to you)")
        print("3. Make sure the camera has a clear view of the entire board")
        print("")
        print("Controls:")
        print("  SPACE - Confirm detected board")
        print("  M - Manual corner selection")
        print("  C - Toggle chessboard detection")
        print("  D - Toggle debug view")
        print("  Q - Quit")

        detection_mode = "auto"
        show_debug = False
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_count += 1

            # Detect board corners
            if detection_mode == "chessboard":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = self._detect_chessboard_pattern(gray)
                if corners is None:
                    corners = self.detect_board_corners(frame)
            else:
                corners = self.detect_board_corners(frame)

            # Visual feedback
            if corners is not None:
                self._draw_detection_overlay(frame, corners, detection_mode)
                status_color = (0, 255, 0)
                status_text = f"BOARD DETECTED ({detection_mode})! Press SPACE to confirm"
            else:
                status_color = (0, 0, 255)
                status_text = f"Searching for board ({detection_mode})..."

            # Draw status and controls
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Show coordinate system info
            coord_text = "Fixed: A-H (L→R), 1-8 (B→T), Black on bottom"
            cv2.putText(frame, coord_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            controls = [
                "SPACE: confirm | M: manual | C: chessboard | D: debug | Q: quit",
                f"Frame: {frame_count} | Mode: {detection_mode}"
            ]

            for i, control_text in enumerate(controls):
                cv2.putText(frame, control_text, (10, frame.shape[0] - 40 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show debug view if enabled
            if show_debug:
                debug_frame = self._create_debug_view(frame)
                cv2.imshow('Debug View', debug_frame)

            cv2.imshow('Ultimate Calibration', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and corners is not None:
                # Confirm detection
                self.board_corners = corners
                self.calibrated = True

                # Test perspective extraction
                board_image = self.get_board_perspective(frame, corners)
                if board_image is not None:
                    self.capture_comprehensive_baseline(board_image)
                    cv2.destroyAllWindows()
                    print("✓ Calibration successful!")
                    return True
                else:
                    print("✗ Perspective extraction failed")

            elif key == ord('m'):
                # Manual corner selection
                cv2.destroyAllWindows()
                corners = self.manual_corner_selection()
                if corners is not None:
                    self.board_corners = corners
                    self.calibrated = True

                    # Get a fresh frame and capture baseline
                    ret, frame = self.cap.read()
                    if ret:
                        board_image = self.get_board_perspective(frame, corners)
                        if board_image is not None:
                            self.capture_comprehensive_baseline(board_image)

                    print("✓ Manual calibration successful!")
                    return True
                else:
                    print("✗ Manual calibration failed")
                    return False

            elif key == ord('c'):
                # Toggle detection mode
                detection_mode = "chessboard" if detection_mode == "auto" else "auto"
                print(f"Switched to {detection_mode} detection mode")

            elif key == ord('d'):
                # Toggle debug view
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow('Debug View')

            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

    def _draw_detection_overlay(self, frame: np.ndarray, corners: np.ndarray, mode: str) -> None:
        """Draw detection overlay on frame."""
        # Draw corner points
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)

        # Draw corner numbers
        for i, corner in enumerate(corners_int):
            cv2.circle(frame, tuple(corner), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (corner[0]+10, corner[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw grid preview
        self._draw_grid_overlay(frame, corners)

        # Draw detection method indicator
        cv2.putText(frame, f"Method: {mode}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_grid_overlay(self, frame: np.ndarray, corners: np.ndarray) -> None:
        """Draw chessboard grid overlay."""
        for i in range(1, 8):
            # Vertical lines
            start = corners[0] + (corners[1] - corners[0]) * i / 8
            end = corners[3] + (corners[2] - corners[3]) * i / 8
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 0, 0), 1)

            # Horizontal lines
            start = corners[0] + (corners[3] - corners[0]) * i / 8
            end = corners[1] + (corners[2] - corners[1]) * i / 8
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 0, 0), 1)

    def _create_debug_view(self, frame: np.ndarray) -> np.ndarray:
        """Create debug visualization showing detection process."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        debug_h, debug_w = h // 2, w // 2

        # Create different processed versions
        views = []

        # Original grayscale
        views.append(cv2.resize(gray, (debug_w, debug_h)))

        # Adaptive threshold
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        views.append(cv2.resize(adaptive, (debug_w, debug_h)))

        # Canny edges
        edges = cv2.Canny(blurred, 100, 200)
        views.append(cv2.resize(edges, (debug_w, debug_h)))

        # Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        views.append(cv2.resize(otsu, (debug_w, debug_h)))

        # Combine into 2x2 grid
        top_row = np.hstack([views[0], views[1]])
        bottom_row = np.hstack([views[2], views[3]])
        debug_frame = np.vstack([top_row, bottom_row])

        # Convert to color for labels
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)

        # Add labels
        labels = ["Original", "Adaptive", "Canny", "Otsu"]
        positions = [(5, 20), (debug_w + 5, 20), (5, debug_h + 20), (debug_w + 5, debug_h + 20)]

        for label, pos in zip(labels, positions):
            cv2.putText(debug_frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return debug_frame

    def manual_corner_selection(self) -> Optional[np.ndarray]:
        """Enhanced manual corner selection with visual guidance."""
        print("\n=== MANUAL CORNER SELECTION ===")
        print("Click the corners of your chessboard in this exact order:")
        print("1. TOP-LEFT corner")
        print("2. TOP-RIGHT corner")
        print("3. BOTTOM-RIGHT corner")
        print("4. BOTTOM-LEFT corner")
        print("Press 'r' to reset, 'q' to quit")

        corners = []
        corner_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                print(f"✓ {corner_names[len(corners)-1]} selected: ({x}, {y})")

        cv2.namedWindow('Manual Corner Selection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Manual Corner Selection', mouse_callback)

        while len(corners) < 4:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Draw existing corners
            for i, corner in enumerate(corners):
                cv2.circle(frame, tuple(corner), 10, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (corner[0]+15, corner[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw connections between corners
            if len(corners) > 1:
                for i in range(len(corners)):
                    if i < len(corners) - 1:
                        cv2.line(frame, tuple(corners[i]), tuple(corners[i+1]), (255, 0, 0), 2)
                    elif len(corners) == 4:
                        cv2.line(frame, tuple(corners[3]), tuple(corners[0]), (255, 0, 0), 2)

            # Instructions
            next_corner = len(corners)
            if next_corner < 4:
                instruction = f"Click {corner_names[next_corner]} corner ({next_corner+1}/4)"
                cv2.putText(frame, instruction, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, "R: reset | Q: quit",
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Manual Corner Selection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('Manual Corner Selection')
                return None
            elif key == ord('r'):
                corners.clear()
                print("Reset - select corners again")

        cv2.destroyWindow('Manual Corner Selection')

        if len(corners) == 4:
            return self.order_points(np.array(corners, dtype="float32"))

        return None

    # ===== VISUALIZATION =====

    def update_visualization(self) -> None:
        """Update PyGame visualization if enabled."""
        if not self.enable_visualization or not self.pygame_initialized:
            return

        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Clear screen
        self.pygame_screen.fill((240, 217, 181))  # Light wood color

        # Draw chessboard
        self._draw_chessboard()

        # Draw pieces
        if self.current_fen:
            self._draw_pieces()

        # Draw info panel
        self._draw_info_panel()

        pygame.display.flip()
        return True

    def _draw_chessboard(self) -> None:
        """Draw the chessboard squares with coordinate labels."""
        light_color = (240, 217, 181)
        dark_color = (181, 136, 99)

        for row in range(8):
            for col in range(8):
                color = light_color if (row + col) % 2 == 0 else dark_color

                rect = pygame.Rect(col * self.square_size, row * self.square_size,
                                 self.square_size, self.square_size)
                pygame.draw.rect(self.pygame_screen, color, rect)

                # Draw square coordinate labels
                if self.show_square_recognition:
                    square_name = f"{self.files[col]}{self.rank_numbers[row]}"

                    # Choose text color based on square color
                    text_color = (100, 80, 60) if (row + col) % 2 == 0 else (200, 180, 140)

                    text = self.small_font.render(square_name, True, text_color)
                    text_rect = pygame.Rect(col * self.square_size + 2,
                                          row * self.square_size + 2,
                                          text.get_width(), text.get_height())
                    self.pygame_screen.blit(text, text_rect)

        # Draw border
        border_rect = pygame.Rect(0, 0, self.board_size, self.board_size)
        pygame.draw.rect(self.pygame_screen, (0, 0, 0), border_rect, 2)

    def _draw_pieces(self) -> None:
        """Draw pieces on the board."""
        board_array = self.fen_to_board_array(self.current_fen.split()[0])

        piece_to_image = {
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk',
            'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk'
        }

        for row in range(8):
            for col in range(8):
                piece = board_array[row][col]
                if piece != '.':
                    x = col * self.square_size
                    y = row * self.square_size

                    if self.piece_images and piece in piece_to_image:
                        piece_key = piece_to_image[piece]
                        if piece_key in self.piece_images:
                            self.pygame_screen.blit(self.piece_images[piece_key], (x, y))
                    else:
                        # Text fallback
                        text = self.font.render(piece, True, (0, 0, 0))
                        text_rect = text.get_rect(center=(x + self.square_size//2, y + self.square_size//2))
                        self.pygame_screen.blit(text, text_rect)

    def _draw_info_panel(self) -> None:
        """Draw information panel."""
        info_x = self.board_size + 10
        y_offset = 10

        info_lines = [
            "Ultimate Chess Detector",
            "",
            "FIXED COORDINATE SYSTEM:",
            "X: A-H (left→right)",
            "Y: 1-8 (bottom→top)",
            "Black pieces on bottom",
            "",
            f"Status: {'Calibrated' if self.calibrated else 'Not Calibrated'}",
            f"Detection: {'Active' if self.baseline_captured else 'Learning'}",
            f"Square Display: {'ON' if self.show_square_recognition else 'OFF'}",
            "",
            "Current FEN:",
            self._wrap_text(self.current_fen, 30),
            "",
            f"Stability: {len(self.detection_history)}/{self.stability_frames}",
            "",
            "Last Move:",
            self._wrap_text(self.last_move_description or "None", 30),
            "",
            "Move History:",
        ]

        # Add recent moves to display
        recent_moves = list(self.move_history)[-3:]  # Show last 3 moves
        for i, move in enumerate(recent_moves):
            move_text = f"{len(self.move_history)-len(recent_moves)+i+1}. {self._wrap_text(move, 25)}"
            info_lines.append(move_text)

        info_lines.extend([
            "",
            "Controls:",
            "Q - Quit",
            "R - Reset position",
            "S - Save templates",
            "T - Toggle methods",
            "V - Toggle squares",
            "H - Move history",
            "B - Board state",
            "X - Square analysis"
        ])

        for line in info_lines:
            if line:  # Skip empty lines
                text = self.small_font.render(line, True, (0, 0, 0))
                self.pygame_screen.blit(text, (info_x, y_offset))
            y_offset += 18

    def _wrap_text(self, text: str, max_length: int) -> str:
        """Wrap text to specified length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

    # ===== TEMPLATE MANAGEMENT =====

    def save_templates(self, filename: Optional[str] = None) -> None:
        """Save learned templates and baseline data."""
        if not self.baseline_captured:
            print("No baseline data to save")
            return

        if filename is None:
            timestamp = int(time.time())
            filename = f"chess_templates_{timestamp}.pkl"

        template_data = {
            'version': '2.0',
            'timestamp': time.time(),
            'coordinate_system': 'FIXED: A-H (left-right), 1-8 (bottom-top), Black on bottom',
            'baseline_data': self.baseline_data,
            'piece_templates': self.piece_templates,
            'empty_templates': self.empty_templates,
            'lighting_baseline': self.lighting_baseline,
            'starting_fen': self.starting_fen
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(template_data, f)
            print(f"✓ Templates saved to {filename}")
        except Exception as e:
            print(f"✗ Error saving templates: {e}")

    def load_templates(self, filename: Optional[str] = None) -> bool:
        """Load templates and baseline data."""
        if filename is None:
            # Find most recent template file
            template_files = [f for f in os.listdir('.')
                            if f.startswith('chess_templates_') and f.endswith('.pkl')]

            if not template_files:
                print("No template files found")
                return False

            filename = max(template_files, key=lambda x: os.path.getctime(x))

        try:
            with open(filename, 'rb') as f:
                template_data = pickle.load(f)

            # Check template compatibility
            if 'coordinate_system' in template_data:
                print(f"Template coordinate system: {template_data['coordinate_system']}")
            elif template_data.get('white_on_bottom') is not None:
                # Old format template - check if compatible
                old_white_on_bottom = template_data.get('white_on_bottom')
                if old_white_on_bottom:
                    print("Warning: Template was created with white-on-bottom orientation")
                    print("Current system uses black-on-bottom. Template may not work correctly.")
                else:
                    print("Template orientation matches current system (black on bottom)")

            self.baseline_data = template_data['baseline_data']
            self.piece_templates = template_data['piece_templates']
            self.empty_templates = template_data['empty_templates']
            self.lighting_baseline = template_data['lighting_baseline']
            self.baseline_captured = True

            print(f"✓ Templates loaded from {filename}")
            return True

        except Exception as e:
            print(f"✗ Error loading templates: {e}")
            return False

    # ===== MAIN DETECTION LOOP =====

    def run(self) -> None:
        """Main detection loop with ultimate accuracy."""
        # Calibration
        if not self.calibrate():
            print("Calibration failed. Exiting.")
            return

        # Initialize visualization if enabled
        if self.enable_visualization:
            self.initialize_visualization()

        print("\n=== ULTIMATE CHESS DETECTION ACTIVE ===")
        print(f"FIXED COORDINATE SYSTEM:")
        print(f"  X axis: A-H (left to right)")
        print(f"  Y axis: 1-8 (bottom to top)")
        print(f"  Black pieces on bottom (ranks 1-2)")
        print(f"Starting position: {self.starting_fen}")
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset to starting position")
        print("  S - Save learned templates")
        print("  L - Load templates")
        print("  T - Toggle detection methods")
        print("  V - Toggle square recognition display")
        print("  H - Show complete move history")
        print("  B - Show current board state")
        print("  X - Show square-by-square analysis")
        if self.enable_visualization:
            print("  PyGame window shows live position")
        print("")

        frame_count = 0
        last_report_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame_count += 1

                # Get current FEN
                current_fen = self.get_current_fen(frame)
                if current_fen is None:
                    continue

                # Add to detection history
                self.detection_history.append(current_fen)

                # Check for stable consensus
                if len(self.detection_history) >= self.stability_frames:
                    recent_detections = list(self.detection_history)[-self.stability_frames:]
                    detection_counts = Counter(recent_detections)
                    most_common = detection_counts.most_common(1)[0]

                    # Require strong consensus
                    required_consensus = max(int(self.stability_frames * self.consensus_threshold),
                                           self.stability_frames - 3)

                    if most_common[1] >= required_consensus:
                        stable_fen = most_common[0]

                        if stable_fen != self.last_stable_fen:
                            if self.validate_fen_transition(stable_fen):
                                # Analyze what move was made
                                move_description = self.analyze_move(self.last_stable_fen, stable_fen)

                                self.current_fen = stable_fen

                                # Store move information
                                if move_description:
                                    self.last_move_description = move_description
                                    self.move_history.append(move_description)
                                    print(f"MOVE DETECTED: {move_description}")
                                    print(f"NEW POSITION: {stable_fen}")
                                else:
                                    print(f"POSITION CHANGE: {stable_fen}")

                                self.last_stable_fen = stable_fen
                            else:
                                # Invalid transition - clear some history to recover
                                self.detection_history = deque(
                                    list(self.detection_history)[-5:],
                                    maxlen=30
                                )

                # Update visualization
                if self.enable_visualization:
                    if not self.update_visualization():
                        break

                # Draw camera view with minimal overlay
                if frame_count % 2 == 0:  # Reduce drawing frequency
                    self._draw_camera_overlay(frame, frame_count)
                    cv2.imshow('Ultimate Chess Detector', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_position()
                elif key == ord('s'):
                    self.save_templates()
                elif key == ord('l'):
                    self.load_templates()
                elif key == ord('t'):
                    self._toggle_detection_methods()
                elif key == ord('v'):
                    self._toggle_square_recognition()
                elif key == ord('h'):
                    self._show_move_history()
                elif key == ord('b'):
                    self._show_current_board_state()
                elif key == ord('x'):
                    self._show_square_analysis()
                elif key == ord('t'):
                    self._toggle_detection_methods()

                # Periodic status report
                current_time = time.time()
                if current_time - last_report_time > 30:  # Every 30 seconds
                    self._print_status_report(frame_count, current_time - last_report_time)
                    last_report_time = current_time
                    frame_count = 0

        except KeyboardInterrupt:
            print("\nStopping detection...")

        finally:
            self.cleanup()

    def _draw_square_recognition_overlay(self, frame: np.ndarray) -> None:
        """Draw blue dots and labels in each square to show recognition."""
        if self.board_corners is None:
            return

        # Calculate square positions in perspective
        for row in range(8):
            for col in range(8):
                # Calculate square center in board coordinates (0-800)
                square_size = 100  # 800 / 8
                center_x = col * square_size + square_size // 2
                center_y = row * square_size + square_size // 2

                # Convert board coordinates to camera coordinates
                board_point = np.array([center_x, center_y], dtype=np.float32)

                # Transform from board space to camera space
                # Create transformation matrix from board to camera
                dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(dst, self.board_corners)

                # Transform the point
                board_point_homogeneous = np.array([[[center_x, center_y]]], dtype=np.float32)
                camera_point = cv2.perspectiveTransform(board_point_homogeneous, matrix)

                camera_x = int(camera_point[0][0][0])
                camera_y = int(camera_point[0][0][1])

                # Draw blue dot
                cv2.circle(frame, (camera_x, camera_y), 8, (255, 0, 0), -1)  # Blue dot (BGR format)
                cv2.circle(frame, (camera_x, camera_y), 8, (255, 255, 255), 2)  # White border

                # Draw square label (chess notation)
                square_name = f"{self.files[col]}{self.rank_numbers[row]}"

                # Calculate text position (slightly offset from dot)
                text_x = camera_x - 15
                text_y = camera_y - 15

                # Make sure text stays in frame
                text_x = max(10, min(text_x, frame.shape[1] - 40))
                text_y = max(20, min(text_y, frame.shape[0] - 10))

                # Draw text with background for visibility
                cv2.putText(frame, square_name, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black background
                cv2.putText(frame, square_name, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

    def _draw_camera_overlay(self, frame: np.ndarray, frame_count: int) -> None:
        """Draw minimal overlay on camera view with square recognition."""
        if self.board_corners is not None:
            # Draw board outline
            corners = self.board_corners.astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 3)

            # Draw square recognition dots and labels
            if self.calibrated and self.show_square_recognition:
                self._draw_square_recognition_overlay(frame)

        # Status indicators
        status_lines = [
            f"Ultimate Chess Detector - Frame {frame_count}",
            f"Status: {'Tracking' if self.baseline_captured else 'Learning'}",
        ]

        if len(self.detection_history) > 0:
            consensus = len([fen for fen in self.detection_history
                           if fen == self.detection_history[-1]])
            status_lines.append(f"Stability: {consensus}/{self.stability_frames}")

        # Add last move information
        if self.last_move_description:
            # Truncate long move descriptions for display
            move_text = self.last_move_description
            if len(move_text) > 50:
                move_text = move_text[:47] + "..."
            status_lines.append(f"Last Move: {move_text}")

        # Draw status with background for better visibility
        for i, line in enumerate(status_lines):
            y_pos = 30 + i * 25
            # Black background
            cv2.putText(frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            # Green text
            cv2.putText(frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Controls
        controls_text = "Q:quit R:reset S:save L:load T:toggle V:squares H:history B:board X:analysis"
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black background
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Yellow text

    def _reset_position(self) -> None:
        """Reset to starting position."""
        self.current_fen = self.starting_fen
        self.last_stable_fen = self.starting_fen
        self.detection_history.clear()
        self.last_move_description = None
        self.move_history.clear()
        print(f"RESET TO: {self.starting_fen}")
        print("Move history cleared")

    def _show_square_analysis(self) -> None:
        """Display detailed square-by-square analysis."""
        self.print_square_by_square_analysis(self.current_fen)

    def _show_current_board_state(self) -> None:
        """Display the current board state in detail."""
        self.print_board_state(self.current_fen, "CURRENT BOARD STATE")

    def _show_move_history(self) -> None:
        """Display the complete move history."""
        print("\n=== MOVE HISTORY ===")
        if not self.move_history:
            print("No moves recorded yet")
        else:
            for i, move in enumerate(self.move_history, 1):
                print(f"{i}. {move}")
        print("===================\n")

    def _toggle_square_recognition(self) -> None:
        """Toggle square recognition display."""
        self.show_square_recognition = not self.show_square_recognition
        status = "ON" if self.show_square_recognition else "OFF"
        print(f"Square recognition display: {status}")

    def _toggle_detection_methods(self) -> None:
        """Toggle detection methods on/off."""
        print("\nDetection Methods:")
        for i, (method, enabled) in enumerate(self.detection_methods.items()):
            print(f"{i+1}. {method}: {'ON' if enabled else 'OFF'}")

        try:
            choice = input("Enter method number to toggle (1-5): ").strip()
            if choice.isdigit():
                method_index = int(choice) - 1
                method_names = list(self.detection_methods.keys())
                if 0 <= method_index < len(method_names):
                    method = method_names[method_index]
                    self.detection_methods[method] = not self.detection_methods[method]
                    print(f"Toggled {method}: {'ON' if self.detection_methods[method] else 'OFF'}")
        except Exception as e:
            print(f"Invalid input: {e}")

    def _print_status_report(self, frames_processed: int, elapsed_time: float) -> None:
        """Print periodic status report."""
        fps = frames_processed / elapsed_time if elapsed_time > 0 else 0

        print(f"\n--- STATUS REPORT ---")
        print(f"FPS: {fps:.1f}")
        print(f"Detection history: {len(self.detection_history)} frames")
        print(f"Current position: {self.current_fen[:50]}...")

        if len(self.detection_history) > 0:
            recent_stability = Counter(list(self.detection_history)[-10:])
            most_common = recent_stability.most_common(1)[0]
            stability_pct = (most_common[1] / min(10, len(self.detection_history))) * 100
            print(f"Recent stability: {stability_pct:.1f}%")

        print("--- END REPORT ---\n")

    def cleanup(self) -> None:
        """Clean up all resources."""
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        if self.enable_visualization and self.pygame_initialized:
            pygame.quit()

        print("Cleanup completed")

    def set_board_corners(self, corners):
        """Set board corners from a list of 4 dicts with x/y keys (from frontend)."""
        import numpy as np
        if len(corners) == 4:
            self.board_corners = np.array([[c['x'], c['y']] for c in corners], dtype=np.float32)
            self.calibrated = True
            print(f"Board corners set externally: {self.board_corners}")
        else:
            print("Invalid corners received for calibration.")


def test_camera(camera_index: int = 0) -> bool:
    """
    Test camera functionality before running the main program.
    Returns True if camera works, False otherwise.
    """
    print(f"\n=== CAMERA TEST ===")
    print(f"Testing camera {camera_index}...")

    # List all available cameras first
    print("\nScanning for available cameras...")
    available_cameras = []

    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"  Camera {i}: ✓ Working ({frame.shape})")
                else:
                    print(f"  Camera {i}: ✗ Opens but no frames")
                cap.release()
        except Exception as e:
            print(f"  Camera {i}: ✗ Error - {e}")

    if not available_cameras:
        print("\n❌ No working cameras found!")
        print("\nTroubleshooting steps:")
        print("1. Check camera permissions in system settings")
        print("2. Close other applications using the camera")
        print("3. Try connecting a different camera")
        print("4. Restart your computer")
        return False

    print(f"\n✅ Found {len(available_cameras)} working camera(s): {available_cameras}")

    if camera_index not in available_cameras:
        print(f"\n⚠️  Requested camera {camera_index} not working.")
        print(f"Suggested alternatives: {available_cameras}")
        return False

    # Test the specific camera
    print(f"\nTesting camera {camera_index} in detail...")

    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return False

        # Test frame capture
        successful_frames = 0
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                successful_frames += 1
            time.sleep(0.1)

        cap.release()

        if successful_frames >= 8:
            print(f"✅ Camera {camera_index} is working well! ({successful_frames}/10 frames)")
            return True
        else:
            print(f"⚠️  Camera {camera_index} is unstable ({successful_frames}/10 frames)")
            return False

    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def main():
    """Main function with enhanced argument parsing and camera testing."""
    parser = argparse.ArgumentParser(description='Ultimate Chess Move Detector - FIXED COORDINATE SYSTEM')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--visualization', action='store_true',
                       help='Enable PyGame visualization')
    parser.add_argument('--load-templates', type=str,
                       help='Load templates from file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--test-camera', action='store_true',
                       help='Test camera functionality and exit')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available cameras and exit')

    args = parser.parse_args()

    print("=== ULTIMATE CHESS MOVE DETECTOR ===")
    print("FIXED COORDINATE SYSTEM - Always the same layout!")
    print("  • X axis: A-H (left to right)")
    print("  • Y axis: 1-8 (bottom to top)")
    print("  • Black pieces always on bottom")
    print("  • No orientation confusion!")

    # Note about pygame warnings (they're harmless)
    if PYGAME_AVAILABLE:
        print("Note: PyGame deprecation warnings are harmless and can be ignored")

    print("")

    # Handle special commands
    if args.list_cameras:
        print("Scanning for available cameras...")
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera {i}: ✅ Working ({frame.shape})")
                    else:
                        print(f"Camera {i}: ⚠️  Opens but no frames")
                    cap.release()
                else:
                    print(f"Camera {i}: ❌ Cannot open")
            except Exception as e:
                print(f"Camera {i}: ❌ Error - {e}")
        return

    if args.test_camera:
        success = test_camera(args.camera)
        if success:
            print(f"\n🎉 Camera {args.camera} is ready for chess detection!")
            response = input("\nProceed with chess detection? (y/n): ")
            if response.lower() != 'y':
                return
        else:
            print(f"\n❌ Camera {args.camera} failed tests. Please fix camera issues first.")
            return

    # Create detector with FIXED coordinate system
    detector = UltimateChessDetector(
        enable_visualization=args.visualization
    )

    try:
        # Initialize camera
        detector.initialize_camera(args.camera)

        # Load templates if specified
        if args.load_templates:
            detector.load_templates(args.load_templates)

        # Run detection
        detector.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

        # Provide helpful suggestions based on the error
        if "camera" in str(e).lower() or "frame" in str(e).lower():
            print("\n💡 Camera-related error detected!")
            print("Try these commands:")
            print("  python ultimate_chess_detector.py --test-camera")
            print("  python ultimate_chess_detector.py --list-cameras")
            print("  python ultimate_chess_detector.py --camera 1  # Try different camera")

        print("\n💡 Remember: This detector uses a FIXED coordinate system")
        print("  Set up your board with BLACK pieces on the BOTTOM")
        print("  A1 = bottom-left, H8 = top-right")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
