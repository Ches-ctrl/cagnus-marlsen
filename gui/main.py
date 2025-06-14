from tensorflow.keras.models import load_model

from_model = load_model('models/1200-elo/from.h5', compile=False)
to_model = load_model('models/1200-elo/to.h5', compile=False)

import pygame 
import chess 

from players import HumanPlayer, AIPlayer
from draw import draw_background, draw_pieces
from fen_history import FenHistory
import globals

pygame.init()

SCREEN_WIDTH = 700
SCREEN_HEIGHT = 600

win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Chess')

board = chess.Board()

# Create FEN history instance
fen_history = FenHistory()

white = HumanPlayer(colour='white')
white_ai = AIPlayer(colour='white', from_model=from_model, to_model=to_model)
black_ai = AIPlayer(colour='black', from_model=from_model, to_model=to_model)
black = black_ai

fps_clock = pygame.time.Clock()

run = True 
white_move = True
human_white = True
game_over_countdown = 50

def reset():
	board.reset()
	global white_move
	white_move = True 

	globals.from_square = None 
	globals.to_square = None
	
	# Clear FEN history when resetting game
	fen_history.clear_history()
	# Add starting position to history
	fen_history.add_fen(board.fen())

# Add starting position to history
fen_history.add_fen(board.fen())

while run:
	fps_clock.tick(30)

	draw_background(win=win)
	draw_pieces(win=win, fen=board.fen(), human_white=human_white)

	pygame.display.update()

	if board.is_game_over():
		if game_over_countdown > 0:
			game_over_countdown -= 1
		else:
			# Print final game history before reset
			print("\n" + "="*60)
			print("GAME OVER - Final FEN History:")
			print(fen_history.get_formatted_history())
			print("="*60 + "\n")
			
			reset()
			game_over_countdown = 50
		continue

	if white_move and not human_white:
		white.move(board=board, human_white=human_white, fen_history=fen_history)
		white_move = not white_move

	if not white_move and human_white:
		black.move(board=board, human_white=human_white, fen_history=fen_history)
		white_move = not white_move

	events = pygame.event.get()

	for event in events:
		if event.type == pygame.QUIT:
			run = False
			pygame.quit()

		elif event.type == pygame.MOUSEBUTTONDOWN:
			x, y = pygame.mouse.get_pos()

			if 625 <= x <= 675 and 200 <= y <= 260: # Change sides
				human_white = not human_white

				if human_white:
					white = black 
					white.colour = 'white'
					black = black_ai 
				else:
					black = white 
					black.colour = 'black'
					white = white_ai

				reset()

			elif 630 <= x <= 670 and 320 <= y <= 360: # Reset
				reset()
		
		if white_move and human_white and white.move(board=board, event=event, human_white=human_white, fen_history=fen_history):
			white_move = not white_move
		
		elif not white_move and not human_white and black.move(board=board, event=event, human_white=human_white, fen_history=fen_history):
			white_move = not white_move
			