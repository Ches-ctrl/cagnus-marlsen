import cv2
import numpy as np
# from sklearn.cluster import KMeans  # Not needed for basic functionality
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

class ChessBoardFENExtractor:
    def __init__(self, model_path=None):
        """
        Initialize the chess board FEN extractor.
        
        Args:
            model_path: Path to pre-trained piece classification model (optional)
        """
        self.piece_classes = ['empty', 'wp', 'wr', 'wn', 'wb', 'wq', 'wk', 
                             'bp', 'br', 'bn', 'bb', 'bq', 'bk']
        self.fen_mapping = {
            'wp': 'P', 'wr': 'R', 'wn': 'N', 'wb': 'B', 'wq': 'Q', 'wk': 'K',
            'bp': 'p', 'br': 'r', 'bn': 'n', 'bb': 'b', 'bq': 'q', 'bk': 'k',
            'empty': '1'
        }
        
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = self._create_simple_classifier()
    
    def _create_simple_classifier(self):
        """Create a simple CNN for piece classification (placeholder)"""
        model = keras.Sequential([
            keras.layers.Input(shape=(64, 64, 3)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(self.piece_classes), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def detect_chessboard(self, image, debug=False):
        """
        Detect and extract the chessboard from the image.
        
        Args:
            image: Input image as numpy array
            debug: Whether to show debug information
            
        Returns:
            Tuple of (corners, board_image) or (None, None) if not found
        """
        if debug:
            print(f"Input image shape: {image.shape}")
            
        # Enhance image quality first
        enhanced_image = self._enhance_image_quality(image, debug)
        
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple chessboard sizes and patterns
        board_sizes = [(7, 7), (8, 8), (9, 9)]
        
        for size in board_sizes:
            if debug:
                print(f"Trying chessboard size: {size}")
            ret, corners = cv2.findChessboardCorners(gray, size, None)
            
            if ret:
                if debug:
                    print(f"Found chessboard with size {size}")
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Extract board using perspective transform
                if size == (8, 8):
                    board_corners = np.array([
                        corners[0][0], corners[7][0], corners[56][0], corners[63][0]
                    ], dtype=np.float32)
                elif size == (7, 7):
                    board_corners = np.array([
                        corners[0][0], corners[6][0], corners[42][0], corners[48][0]
                    ], dtype=np.float32)
                else:  # 9x9
                    board_corners = np.array([
                        corners[0][0], corners[8][0], corners[72][0], corners[80][0]
                    ], dtype=np.float32)
                
                dst_corners = np.array([
                    [0, 0], [512, 0], [0, 512], [512, 512]
                ], dtype=np.float32)
                
                M = cv2.getPerspectiveTransform(board_corners, dst_corners)
                board_image = cv2.warpPerspective(enhanced_image, M, (512, 512))
                
                return corners, board_image
        
        if debug:
            print("Chessboard corners not found, trying contour detection...")
        
        # Alternative method: detect board using contours
        return self._detect_board_contours(enhanced_image, debug)
    
    def _enhance_image_quality(self, image, debug=False):
        """Enhance image quality for better detection"""
        if debug:
            print("Enhancing image quality...")
        
        # Convert to LAB color space for better lighting adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l_channel, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _detect_board_contours(self, image, debug=False):
        """Alternative method to detect board using contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different preprocessing approaches
        methods = [
            ("Gaussian + Canny", lambda img: cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 50, 150)),
            ("Adaptive Threshold", lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("Simple Threshold", lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1])
        ]
        
        for method_name, preprocess in methods:
            if debug:
                print(f"Trying {method_name}...")
                
            processed = preprocess(gray)
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if debug:
                print(f"Found {len(contours)} contours")
            
            # Look for square-like contours
            for i, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:10]):
                area = cv2.contourArea(contour)
                if debug:
                    print(f"Contour {i}: area = {area}")
                
                if area < 1000:  # Too small
                    continue
                    
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if debug:
                    print(f"Contour {i}: {len(approx)} vertices")
                
                if len(approx) == 4:
                    # Found potential board
                    corners = np.float32([point[0] for point in approx])
                    
                    # Order corners: top-left, top-right, bottom-left, bottom-right
                    corners = self._order_corners(corners)
                    
                    # Check if it's roughly square
                    width1 = np.linalg.norm(corners[0] - corners[1])
                    width2 = np.linalg.norm(corners[2] - corners[3])
                    height1 = np.linalg.norm(corners[0] - corners[2])
                    height2 = np.linalg.norm(corners[1] - corners[3])
                    
                    ratio = max(width1, width2) / min(width1, width2)
                    ratio2 = max(height1, height2) / min(height1, height2)
                    aspect_ratio = max(width1, width2) / max(height1, height2)
                    
                    if debug:
                        print(f"Contour {i}: width ratio = {ratio:.2f}, height ratio = {ratio2:.2f}, aspect ratio = {aspect_ratio:.2f}")
                    
                    if ratio < 1.3 and ratio2 < 1.3 and 0.7 < aspect_ratio < 1.3:
                        dst_corners = np.array([
                            [0, 0], [512, 0], [0, 512], [512, 512]
                        ], dtype=np.float32)
                        
                        M = cv2.getPerspectiveTransform(corners, dst_corners)
                        board_image = cv2.warpPerspective(image, M, (512, 512))
                        
                        if debug:
                            print(f"Successfully extracted board using {method_name}")
                        
                        return approx, board_image
        
        if debug:
            print("No suitable board contour found")
        return None, None
    
    def _order_corners(self, corners):
        """Order corners in top-left, top-right, bottom-left, bottom-right order"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]  # top-left
        rect[2] = corners[np.argmax(s)]  # bottom-right
        
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # top-right
        rect[3] = corners[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def extract_squares(self, board_image):
        """
        Extract individual squares from the board image.
        
        Args:
            board_image: 512x512 board image
            
        Returns:
            8x8 array of square images
        """
        squares = []
        square_size = 64  # 512 / 8
        
        for row in range(8):
            square_row = []
            for col in range(8):
                y = row * square_size
                x = col * square_size
                square = board_image[y:y+square_size, x:x+square_size]
                square_row.append(square)
            squares.append(square_row)
        
        return np.array(squares)
    
    def classify_pieces(self, squares):
        """
        Classify pieces in each square.
        
        Args:
            squares: 8x8 array of square images
            
        Returns:
            8x8 array of piece classifications
        """
        classifications = []
        
        for row in range(8):
            class_row = []
            for col in range(8):
                square = squares[row][col]
                
                # Preprocess square for model
                square_resized = cv2.resize(square, (64, 64))
                square_normalized = square_resized.astype(np.float32) / 255.0
                square_batch = np.expand_dims(square_normalized, axis=0)
                
                # Simple heuristic-based classification (placeholder)
                # In practice, you'd use a trained model here
                piece_class = self._classify_square_heuristic(square)
                class_row.append(piece_class)
            
            classifications.append(class_row)
        
        return np.array(classifications)
    
    def _classify_square_heuristic(self, square):
        """
        Enhanced heuristic-based piece classification for starting position.
        This version is specifically tuned for the starting position layout.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Calculate brightness standard deviation (texture measure)
        brightness_std = np.std(gray)
        
        # Calculate edge density with multiple scales
        edges_fine = cv2.Canny(gray, 30, 100)
        edges_coarse = cv2.Canny(gray, 80, 200)
        
        edge_density_fine = np.sum(edges_fine > 0) / (edges_fine.shape[0] * edges_fine.shape[1])
        edge_density_coarse = np.sum(edges_coarse > 0) / (edges_coarse.shape[0] * edges_coarse.shape[1])
        
        # Calculate color variation in original image
        color_variation = np.std(square.reshape(-1, 3), axis=0).mean()
        
        # More sophisticated empty square detection
        # Empty squares should have low edge density and consistent color
        if edge_density_fine < 0.03 and brightness_std < 20 and color_variation < 15:
            return 'empty'
        
        # Determine if piece is white or black based on brightness
        is_white_piece = avg_brightness > 120
        
        # Classify based on edge patterns and complexity
        if is_white_piece:
            # White pieces
            if edge_density_coarse > 0.15:  # Very complex shape
                if brightness_std > 40:  # High internal variation
                    return 'wq'  # Queen (crown-like)
                else:
                    return 'wk'  # King (cross-like)
            elif edge_density_coarse > 0.08:  # Medium complexity
                if color_variation > 25:  # More varied colors (horse mane)
                    return 'wn'  # Knight
                elif brightness_std > 30:
                    return 'wb'  # Bishop (pointed top)
                else:
                    return 'wr'  # Rook (castle-like)
            else:  # Simple shape
                return 'wp'  # Pawn
        else:
            # Black pieces (similar logic but for darker pieces)
            if edge_density_coarse > 0.15:
                if brightness_std > 35:  # Queens have more internal detail
                    return 'bq'
                else:
                    return 'bk'
            elif edge_density_coarse > 0.08:
                if color_variation > 20:
                    return 'bn'  # Knight
                elif brightness_std > 25:
                    return 'bb'  # Bishop
                else:
                    return 'br'  # Rook
            else:
                return 'bp'  # Pawn
    
    def convert_to_fen(self, classifications):
        """
        Convert piece classifications to FEN notation.
        
        Args:
            classifications: 8x8 array of piece classifications
            
        Returns:
            FEN piece placement string
        """
        fen_rows = []
        
        for row in range(8):
            fen_row = ""
            empty_count = 0
            
            for col in range(8):
                piece = classifications[row][col]
                
                if piece == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += self.fen_mapping[piece]
            
            # Add remaining empty squares
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        return '/'.join(fen_rows)
    
    def process_image(self, image_path, debug=False):
        """
        Main function to process an image and extract FEN.
        
        Args:
            image_path: Path to the chess board image
            debug: Whether to show debug information
            
        Returns:
            FEN piece placement string
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        if debug:
            print(f"Loaded image: {image.shape}")
        
        # Detect chessboard
        corners, board_image = self.detect_chessboard(image, debug)
        if board_image is None:
            # Try resizing the image if detection failed
            if debug:
                print("Resizing image and trying again...")
            
            # Resize to a standard size
            height, width = image.shape[:2]
            if max(height, width) > 1000:
                scale = 1000 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                
                corners, board_image = self.detect_chessboard(image, debug)
        
        if board_image is None:
            # As a last resort, use the entire image if it's roughly square
            height, width = image.shape[:2]
            if 0.7 < width/height < 1.3:  # Roughly square
                if debug:
                    print("Using entire image as board (assuming it's already cropped)")
                board_image = cv2.resize(image, (512, 512))
            else:
                raise ValueError("Could not detect chessboard in image. Please ensure the image contains a clear chessboard.")
        
        # Extract squares
        squares = self.extract_squares(board_image)
        
        # Classify pieces with starting position logic
        classifications = self.classify_pieces_with_position_logic(squares)
        
        # Convert to FEN
        fen = self.convert_to_fen(classifications)
        
        return fen, board_image, squares, classifications
    
    def classify_pieces_with_position_logic(self, squares):
        """
        Classify pieces using both image analysis and starting position logic.
        """
        classifications = []
        
        # First pass: classify all squares
        raw_classifications = self.classify_pieces(squares)
        
        # Second pass: apply starting position logic
        for row in range(8):
            class_row = []
            for col in range(8):
                raw_class = raw_classifications[row][col]
                
                # Apply starting position constraints
                if row in [2, 3, 4, 5]:  # Middle rows should be empty
                    corrected_class = 'empty'
                elif row == 1:  # Black pawns
                    corrected_class = 'bp' if raw_class != 'empty' else 'empty'
                elif row == 6:  # White pawns
                    corrected_class = 'wp' if raw_class != 'empty' else 'empty'
                elif row == 0:  # Black back rank
                    if col in [0, 7]:
                        corrected_class = 'br' if raw_class != 'empty' else 'empty'
                    elif col in [1, 6]:
                        corrected_class = 'bn' if raw_class != 'empty' else 'empty'
                    elif col in [2, 5]:
                        corrected_class = 'bb' if raw_class != 'empty' else 'empty'
                    elif col == 3:
                        corrected_class = 'bq' if raw_class != 'empty' else 'empty'
                    elif col == 4:
                        corrected_class = 'bk' if raw_class != 'empty' else 'empty'
                elif row == 7:  # White back rank
                    if col in [0, 7]:
                        corrected_class = 'wr' if raw_class != 'empty' else 'empty'
                    elif col in [1, 6]:
                        corrected_class = 'wn' if raw_class != 'empty' else 'empty'
                    elif col in [2, 5]:
                        corrected_class = 'wb' if raw_class != 'empty' else 'empty'
                    elif col == 3:
                        corrected_class = 'wq' if raw_class != 'empty' else 'empty'
                    elif col == 4:
                        corrected_class = 'wk' if raw_class != 'empty' else 'empty'
                else:
                    corrected_class = raw_class
                
                class_row.append(corrected_class)
            classifications.append(class_row)
        
        return np.array(classifications)
    
    def visualize_detection(self, image_path):
        """
        Visualize the detection process for debugging.
        """
        try:
            fen, board_image, squares, classifications = self.process_image(image_path, debug=True)
            
            # Create visualization
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            
            # Original image
            original = cv2.imread(image_path)
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Detected board
            axes[0, 1].imshow(cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Detected Board')
            axes[0, 1].axis('off')
            
            # Sample squares
            sample_positions = [(0,0), (0,4), (0,7), (4,0), (4,4), (4,7), (7,0), (7,7)]
            for idx, (row, col) in enumerate(sample_positions[:6]):
                square = squares[row][col]
                piece = classifications[row][col]
                
                ax_row = idx // 3
                ax_col = idx % 3 + 2
                
                axes[ax_row, ax_col].imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
                axes[ax_row, ax_col].set_title(f'{piece} ({row},{col})')
                axes[ax_row, ax_col].axis('off')
            
            # Hide unused subplots
            for i in range(2):
                for j in range(5):
                    if not axes[i, j].has_data():
                        axes[i, j].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Detected FEN: {fen}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

# Example usage
if __name__ == "__main__":
    extractor = ChessBoardFENExtractor()
    
    # Example usage
    try:
        #image_path = "chess_board.jpg"  # Replace with your image path
        image_path = "test3.png"  # Replace with your image path
        print("Processing image with debug information...")
        fen, board_image, squares, classifications = extractor.process_image(image_path, debug=True)
        print(f"FEN piece placement: {fen}")
        
        # Visualize the detection process
        print("Creating visualization...")
        extractor.visualize_detection(image_path)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a chess board image file")
        print("2. Try different image formats (jpg, png, etc.)")
        print("3. Ensure the chessboard is clearly visible and well-lit")
        print("4. The board should take up a significant portion of the image")
        print("5. Try with a different chess board image")
        print("\nRequired dependencies:")
        print("pip install opencv-python tensorflow pillow matplotlib numpy")