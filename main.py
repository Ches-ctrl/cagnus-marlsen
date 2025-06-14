import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_board_image(image_path):
    """
    Analyze a board image using OpenCV - detects edges, contours, and grid structure
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Successfully loaded image: {image_path}")
    print(f"Image dimensions: {img.shape}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return {
        'original': img,
        'gray': gray,
        'edges': edges,
        'contours': contours
    }

def extract_chess_board(image_path):
    """
    Extract the chess board from the image and divide into 8x8 squares
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the largest rectangular contour (likely the chess board)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in image")
    
    # Find the largest contour (assuming it's the chess board)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Make it square by using the smaller dimension
    size = min(w, h)
    # Center the square
    x_offset = (w - size) // 2
    y_offset = (h - size) // 2
    
    # Extract square board region
    board_roi = img[y + y_offset:y + y_offset + size, x + x_offset:x + x_offset + size]
    
    # Divide into 8x8 grid
    cell_size = size // 8
    
    squares = []
    for row in range(8):
        square_row = []
        for col in range(8):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            
            square = board_roi[y1:y2, x1:x2]
            square_row.append(square)
        squares.append(square_row)
    
    return squares, board_roi

def classify_piece(square, row, col):
    """
    Classify what piece is in a chess square
    Returns: piece character for FEN notation or None for empty
    """
    # Convert to different color spaces for analysis
    gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    hsv_square = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    
    # Calculate brightness statistics
    mean_brightness = np.mean(gray_square)
    std_brightness = np.std(gray_square)
    
    # Check if square is likely empty (low standard deviation suggests uniform color)
    if std_brightness < 20:  # Adjust threshold as needed
        return None  # Empty square
    
    # Detect if there's a piece (look for shapes/contours)
    edges = cv2.Canny(gray_square, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours (noise)
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    
    if not significant_contours:
        return None  # No significant shapes found
    
    # Determine piece color based on brightness
    # This is a simple heuristic - you may need to adjust based on your image
    piece_area_mask = np.zeros_like(gray_square)
    cv2.drawContours(piece_area_mask, significant_contours, -1, 255, -1)
    piece_pixels = gray_square[piece_area_mask > 0]
    
    if len(piece_pixels) == 0:
        return None
    
    avg_piece_brightness = np.mean(piece_pixels)
    
    # Determine if piece is white or black
    is_white = avg_piece_brightness > 128  # Adjust threshold as needed
    
    # Piece shape analysis for type detection
    largest_contour = max(significant_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        return None
    
    # Shape features
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Get bounding rectangle aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Simple piece classification based on shape features
    # This is a basic heuristic - you may want to use machine learning for better accuracy
    piece_type = classify_piece_by_shape(circularity, solidity, aspect_ratio, area)
    
    # Return FEN notation
    if piece_type:
        return piece_type.upper() if is_white else piece_type.lower()
    else:
        # Default to pawn if we detect a piece but can't classify it
        return 'P' if is_white else 'p'

def classify_piece_by_shape(circularity, solidity, aspect_ratio, area):
    """
    Classify piece type based on shape features
    This is a simplified heuristic - real implementation might use ML
    """
    # These thresholds are rough estimates and may need adjustment
    if circularity > 0.7:  # Round shapes
        if area > 1000:  # Larger pieces
            return 'q'  # Queen (large, round crown)
        else:
            return 'p'  # Pawn (small, round top)
    elif solidity > 0.9 and aspect_ratio > 0.8:  # Solid, square-ish
        return 'r'  # Rook (rectangular/solid)
    elif solidity < 0.7:  # Less solid (more complex shape)
        if area > 1200:
            return 'k'  # King (complex crown)
        else:
            return 'n'  # Knight (complex horse shape)
    else:
        return 'b'  # Bishop (pointed top)

def generate_fen_from_board(squares):
    """
    Generate FEN notation from analyzed chess board squares
    """
    fen_rows = []
    
    for row in range(8):
        fen_row = ""
        empty_count = 0
        
        for col in range(8):
            piece = classify_piece(squares[row][col], row, col)
            
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        
        # Add remaining empty squares
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    # Join rows with '/' separator
    board_fen = '/'.join(fen_rows)
    
    # Add additional FEN components (simplified)
    # In a real implementation, you'd need to track these from game state
    active_color = 'w'  # White to move (default)
    castling = 'KQkq'   # All castling available (default)
    en_passant = '-'    # No en passant (default)
    halfmove_clock = '0'  # Halfmove clock
    fullmove_number = '1'  # Move number
    
    full_fen = f"{board_fen} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    
    return board_fen, full_fen

def visualize_board_analysis(squares, board_roi):
    """
    Visualize the board analysis with piece classifications
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Show original board
    axes[0, 0].imshow(cv2.cvtColor(board_roi, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Extracted Chess Board')
    axes[0, 0].axis('off')
    
    # Show some individual squares with their classifications
    for i in range(11):  # Show 11 squares as examples
        row = i // 4
        col = i % 4
        if row < 3 and col < 4:
            if row == 0 and col == 0:
                continue  # Skip first position (used for full board)
            
            # Calculate which chess square to show
            chess_row = (i - 1) // 2
            chess_col = ((i - 1) % 2) * 4
            
            if chess_row < 8 and chess_col < 8:
                square = squares[chess_row][chess_col]
                piece = classify_piece(square, chess_row, chess_col)
                
                axes[row, col].imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
                piece_name = piece if piece else 'Empty'
                axes[row, col].set_title(f'{chr(97+chess_col)}{8-chess_row}: {piece_name}')
                axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(11, 12):
        row = i // 4
        col = i % 4
        if row < 3 and col < 4:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    image_path = "assets/capture_20250614_122339.jpg"
    
    try:
        print(f"Analyzing chess board from: {image_path}")
        
        # Extract chess board squares
        squares, board_roi = extract_chess_board(image_path)
        print("Successfully extracted 8x8 chess board grid")
        
        # Generate FEN notation
        board_fen, full_fen = generate_fen_from_board(squares)
        
        print("\n" + "="*50)
        print("CHESS BOARD ANALYSIS RESULTS")
        print("="*50)
        print(f"Board FEN: {board_fen}")
        print(f"Full FEN:  {full_fen}")
        print("="*50)
        
        # Visualize results
        visualize_board_analysis(squares, board_roi)
        
        return squares, board_fen, full_fen
        
    except Exception as e:
        print(f"Error processing chess board: {e}")
        return None

if __name__ == "__main__":
    main()