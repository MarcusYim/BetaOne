import chess

def main():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("f5")
    board.push_san("g4")
    print(board);
    
    for i in board.legal_moves:
        print(i)
    
if __name__ == "__main__":
    main()
