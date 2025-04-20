import io
import random

import chess
import chess.pgn


def analyze_games(pgn_file = 'lichess_truncated.pgn'):
    # generate dataset annotation dict
    dataset = {}
    dataset['p'] = set() # promotions
    dataset['kc'] = set() # kingside castles
    dataset['qc'] = set() # queenside castles
    count = 0

    # one grouping for each piece count, from 2 (min) to 32 (max)
    for i in range(2, 33):
        dataset['d' + str(i)] = set()

    with open(pgn_file) as f:
        lines = f.readlines()
        print('Analyzing', len(lines), 'games...')
        for line in lines:
            game = chess.pgn.read_game(io.StringIO(line))
            board = game.board()
            for move in game.mainline_moves():
                prev_fen = board.board_fen()
                if board.is_kingside_castling(move):
                    board.push(move)
                    dataset['kc'].add((prev_fen, move.uci(), board.board_fen()))
                elif board.is_queenside_castling(move):
                    board.push(move)
                    dataset['qc'].add((prev_fen, move.uci(), board.board_fen()))
                elif move.promotion is not None:
                    board.push(move)
                    dataset['p'].add((prev_fen, move.uci(), board.board_fen()))
                else:
                    board.push(move)
                    piece_count = len(board.piece_map().values())
                    dataset['d' + str(piece_count)].add((prev_fen, move.uci(), board.board_fen()))
                count += 1

    dataset['total'] = count
    return dataset

def select(dataset, promotions_ratio, kingside_castle_ratio, queenside_castle_ratio, piece_count_ratio, min_count = 0, print_info = False):
    # select the moves
    selected = set()

    toadd = random.sample(sorted(dataset['p']), min(max(min_count, int(promotions_ratio * len(dataset['p']))), len(dataset['p'])))
    selected.update(toadd)

    if print_info:
        print(f"Selected {len(toadd)}/{len(dataset['p'])} promotions.")

    toadd = random.sample(sorted(dataset['kc']), min(max(min_count, int(promotions_ratio * len(dataset['kc']))), len(dataset['kc'])))
    selected.update(toadd)
    
    if print_info:
        print(f"Selected {len(toadd)}/{len(dataset['kc'])} kingside castles.")


    toadd = random.sample(sorted(dataset['qc']), min(max(min_count, int(promotions_ratio * len(dataset['qc']))), len(dataset['qc'])))
    selected.update(toadd)

    if print_info:
        print(f"Selected {len(toadd)}/{len(dataset['qc'])} queenside castles.")

    for i in range(2, 33):
        toadd = random.sample(sorted(dataset['d' + str(i)]), min(max(min_count, int(promotions_ratio * len(dataset['d' + str(i)]))), len(dataset['d' + str(i)])))
        selected.update(toadd)
        if print_info:
            print(f"Selected {len(toadd)}/{len(dataset['d' + str(i)])} moves with {i} pieces.")
    
    if print_info:
        print('Total selected moves:', len(selected))
        
    return selected