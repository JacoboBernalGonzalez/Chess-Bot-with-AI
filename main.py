"""
Script principal para entrenar y probar el bot de ajedrez
"""

import torch
from neural_network import create_model
from trainer import ChessTrainer
from mtcs import MCTS, board_to_tensor
import chess

def train_mode(args):
    """Modo de entrenamiento"""
    print("Iniciando modo de entrenamiento...")
    
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'replay_buffer_size': args.buffer_size,
        'mcts_simulations': args.simulations
    }
    
    # Crear o cargar modelo
    if args.resume:
        print(f"Cargando modelo desde {args.resume}")
        model = create_model(args.model_size, config['device'])
        trainer = ChessTrainer(model, config)
        trainer.load_checkpoint(args.resume)
    else:
        print(f"Creando nuevo modelo: {args.model_size}")
        model = create_model(args.model_size, config['device'])
        trainer = ChessTrainer(model, config)
    
    # Entrenar
    trainer.train_loop(
        iterations=args.iterations,
        games_per_iteration=args.games,
        epochs_per_iteration=args.epochs
    )

def play_mode(args):
    """Modo de juego interactivo"""
    print("Iniciando modo de juego...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(args.model_size, device)
    
    if args.model_path:
        print(f"Cargando modelo desde {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    mcts = MCTS(model, num_simulations=args.simulations)
    board = chess.Board()
    
    print("\nJuego iniciado. Tú juegas con blancas.")
    print("Ingresa movimientos en formato UCI (ej: e2e4)")
    print("Escribe 'exit' para salir\n")
    
    while not board.is_game_over():
        print(board)
        print(f"\nMovimiento #{board.fullmove_number}")
        
        if board.turn == chess.WHITE:
            # Turno del humano
            while True:
                move_str = input("Tu movimiento: ").strip().lower()
                if move_str == 'exit':
                    return
                
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Movimiento ilegal. Intenta de nuevo.")
                except:
                    print("Formato inválido. Usa formato UCI (ej: e2e4)")
        else:
            # Turno del bot
            print("El bot está pensando...")
            move, move_probs = mcts.get_move(board)
            print(f"Bot juega: {move.uci()}")
            
            # Mostrar top movimientos considerados
            print("\nTop movimientos considerados:")
            for move_uci, prob in sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {move_uci}: {prob:.2%}")
            
            board.push(move)
            print()
    
    print("\n" + "="*40)
    print("Juego terminado!")
    print(f"Resultado: {board.result()}")
    print("="*40)

def evaluate_mode(args):
    """Modo de evaluación contra Stockfish u otro engine"""
    print("Modo de evaluación...")
    print("Esta función requiere tener Stockfish instalado")
    print("Implementación pendiente - usa cutechess-cli para torneos")