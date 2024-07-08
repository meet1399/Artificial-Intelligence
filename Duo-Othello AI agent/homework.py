from copy import deepcopy

# Constants
BOARD_SIZE = 12
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = '.'
INF = float('inf')

DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


# Function to read the input file
def read_input_file(file_name):
    with open(file_name, 'r') as file:
        player = file.readline().strip()
        remaining_time, opponent_time = map(float, file.readline().strip().split())
        board = [list(line.strip()) for line in file]
    return player, remaining_time, opponent_time, board

# Function to write the chosen move to the output file
def write_output_file(move, file_name):
    with open(file_name, 'w') as file:
        file.write(move)

# Function to determine legal moves for a player
def legal_moves(player, board):
    moves = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == EMPTY:
                if any(is_valid_move(player, board, i, j, di, dj) for di, dj in DIRECTIONS):
                    moves.append((i, j))
    return moves

# Function to check if a move is valid in any direction
def is_valid_move(player, board, row, col, di, dj):
    opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
    new_row, new_col = row + di, col + dj
    if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE) or board[new_row][new_col] != opponent:
        return False
    new_row += di
    new_col += dj
    while 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
        if board[new_row][new_col] == EMPTY:
            return False
        if board[new_row][new_col] == player:
            return True
        new_row += di
        new_col += dj
    return False

# Function to execute a move and flip opponent's pieces
def execute_move(player, board, row, col):
    board[row][col] = player
    opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
    for dr, dc in DIRECTIONS:
        new_row, new_col = row + dr, col + dc
        flips = []
        while 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE and board[new_row][new_col] == opponent:
            flips.append((new_row, new_col))
            new_row += dr
            new_col += dc
        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE and board[new_row][new_col] == player:
            for flip_row, flip_col in flips:
                board[flip_row][flip_col] = player

# Function to evaluate the current state of the board
def evaluate_board(global_player, board):
    score = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == global_player:
                if (i == 0 or i == BOARD_SIZE - 1) and (j == 0 or j == BOARD_SIZE - 1):
                    # Corner position
                    score += 5
                elif i == 0 or i == BOARD_SIZE - 1 or j == 0 or j == BOARD_SIZE - 1:
                    # Edge position
                    score += 3
                elif has_empty_neighbor(board, i, j):
                    # Empty neighbor
                    score -= 1
                else:
                    score += 1
    return score

# Function to check if any neighboring position of a cell is empty
def has_empty_neighbor(board, row, col):
    for di, dj in DIRECTIONS:
        ni, nj = row + di, col + dj
        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni][nj] == EMPTY:
            return True
    return False

# Minimax function with alpha-beta pruning
def minimax(global_player, player, board, depth, alpha, beta, maximizing_player, pass_flag):
    if depth == 0:
        return evaluate_board(global_player, board), None

    legal_moves_list = legal_moves(player, board)
    if not legal_moves_list:
        if(pass_flag == 1):
            return evaluate_board(global_player, board), None
        maximizing_player = not maximizing_player
        opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
        eval, _ = minimax(global_player, opponent, board, depth, alpha, beta, maximizing_player, 1)
        return eval, None
    elif maximizing_player:
        max_eval = -INF
        best_move = None
        for move in legal_moves_list:
            new_board = deepcopy(board)
            execute_move(player, new_board, move[0], move[1])
            opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
            eval, _ = minimax(global_player, opponent, new_board, depth - 1, alpha, beta, False, 0)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = INF
        best_move = None
        for move in legal_moves_list:
            new_board = deepcopy(board)
            execute_move(player, new_board, move[0], move[1])
            opponent = PLAYER_X if player == PLAYER_O else PLAYER_O
            eval, _ = minimax(global_player, opponent, new_board, depth - 1, alpha, beta, True, 0)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# Main function
def main():
    input_file = "./input.txt"
    output_file = "./output.txt"

    player, remaining_time, opponent_time, board = read_input_file(input_file)

    global_player = player

    # Set depth for minimax search
    if (remaining_time > 296):
        depth = 4
    elif (remaining_time > 210):
        depth = 6
    elif (remaining_time > 150):
        depth = 5
    elif (remaining_time > 30):
        depth = 4

    _, chosen_move = minimax(global_player, player, board, depth, -INF, INF, True, 0)

    if chosen_move is not None:
        move_str = f"{chr(chosen_move[1] + ord('a'))}{chosen_move[0] + 1}"
        # execute_move(player, board, chosen_move[0], chosen_move[1])

    write_output_file(move_str, output_file)

if __name__ == "__main__":
    main()