class FenHistory:
    """
    A class to store and manage FEN (Forsyth-Edwards Notation) values throughout a chess game.
    """
    
    def __init__(self):
        """Initialize with an empty FEN history list."""
        self.fen_list = []
    
    def add_fen(self, fen):
        """
        Add a new FEN string to the history.
        
        Args:
            fen (str): The FEN string to add to the history
        """
        self.fen_list.append(fen)
    
    def clear_history(self):
        """Clear all FEN values from the history."""
        self.fen_list = []
    
    def get_formatted_history(self):
        """
        Get the FEN history as a nicely formatted string.
        
        Returns:
            str: A formatted string showing all FEN positions with move numbers
        """
        if not self.fen_list:
            return "No FEN history available."
        
        formatted_lines = []
        formatted_lines.append("=== CHESS GAME FEN HISTORY ===")
        formatted_lines.append("")
        
        for i, fen in enumerate(self.fen_list):
            # Extract move number and side to move from FEN
            fen_parts = fen.split()
            side_to_move = "White" if fen_parts[1] == 'w' else "Black"
            move_number = fen_parts[5] if len(fen_parts) > 5 else "?"
            
            # Format the entry
            formatted_lines.append(f"Move {i+1:2d} (Turn: {move_number}, {side_to_move} to move):")
            formatted_lines.append(f"  FEN: {fen}")
            formatted_lines.append("")
        
        formatted_lines.append(f"Total positions stored: {len(self.fen_list)}")
        formatted_lines.append("=" * 40)
        
        return "\n".join(formatted_lines)
    
    def get_current_fen(self):
        """
        Get the most recent FEN string.
        
        Returns:
            str: The latest FEN string, or None if history is empty
        """
        return self.fen_list[-1] if self.fen_list else None
    
    def get_fen_count(self):
        """
        Get the number of FEN positions stored.
        
        Returns:
            int: Number of FEN positions in history
        """
        return len(self.fen_list)


# Example usage:
if __name__ == "__main__":
    # Create a FEN history instance
    fen_history = FenHistory()
    
    # Add some example FEN positions
    fen_history.add_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    fen_history.add_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    fen_history.add_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    
    # Print formatted history
    print(fen_history.get_formatted_history())
    
    # Get current FEN
    print(f"\nCurrent FEN: {fen_history.get_current_fen()}")
    
    # Clear and check
    fen_history.clear_history()
    print(f"\nAfter clearing: {fen_history.get_formatted_history()}")