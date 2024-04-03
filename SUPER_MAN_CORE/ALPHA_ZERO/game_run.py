from TicTacToeLogic import Board

# Step 1: Create an instance of the Board class
board = Board()

# Step 2: Play the game
current_player = 1  # White player starts
while board.has_legal_moves():
    # Print the current state of the board
    print("Current board state:")
    for row in board.pieces:
        print(row)

    # Get legal moves for the current player
    legal_moves = board.get_legal_moves(current_player)

    # Check if there are legal moves available
    if not legal_moves:
        print("No legal moves available. Game over.")
        break

    # Let the current player make a move
    print(f"Player {current_player}'s turn.")
    print("Legal moves:", legal_moves)
    x, y = map(int, input("Enter your move (x y): ").split())
    move = (x, y)

    # Execute the move
    board.execute_move(move, current_player)

    # Check if the current player wins
    if board.is_win(current_player):
        print(f"Player {current_player} wins!")
        break

    # Switch player
    current_player *= -1  # Switch between 1 (white) and -1 (black)

