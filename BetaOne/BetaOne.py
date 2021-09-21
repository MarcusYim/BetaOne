import chess

def main():
    board = chess.Board()
    print(board);
    
    for i in board.legal_moves:
        print(i)
    
if __name__ == "__main__":
    main()
