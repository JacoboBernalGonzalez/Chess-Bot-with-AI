"""
Implementación del protocolo UCI para usar el bot con GUIs de ajedrez
y conectarlo a Lichess mediante lichess-bot
"""

import argparse
import sys
import chess
import torch
from neural_network import create_model
from mtcs import MCTS
from main import train_mode, play_mode, evaluate_mode

class UCIEngine:
    """Motor UCI compatible con interfaces gráficas y lichess-bot"""
    def __init__(self, model_path=None, model_size='small', simulations=400):
        self.board = chess.Board()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cargar modelo
        self.model = create_model(model_size, self.device)
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.mcts = MCTS(self.model, num_simulations=simulations, temperature=0.1)
        
        # Info del engine
        self.name = "ChessML-Bot"
        self.author = "AlphaZero Implementation"
    
    def uci(self):
        """Responde al comando uci"""
        print(f"id name {self.name}")
        print(f"id author {self.author}")
        print("option name Simulations type spin default 400 min 50 max 2000")
        print("option name Model type combo default small var small var standard var large")
        print("uciok")
    
    def isready(self):
        """Responde al comando isready"""
        print("readyok")
    
    def position(self, args):
        """Configura la posición del tablero"""
        if args[0] == "startpos":
            self.board = chess.Board()
            moves_idx = args.index("moves") if "moves" in args else len(args)
            moves = args[moves_idx + 1:]
        elif args[0] == "fen":
            moves_idx = args.index("moves") if "moves" in args else len(args)
            fen = " ".join(args[1:moves_idx])
            self.board = chess.Board(fen)
            moves = args[moves_idx + 1:] if "moves" in args else []
        
        # Aplicar movimientos
        for move_uci in moves:
            self.board.push(chess.Move.from_uci(move_uci))
    
    def go(self, args):
        """Busca el mejor movimiento"""
        # Parsear tiempo disponible (simplificado)
        # En implementación completa: gestionar wtime, btime, winc, binc
        
        print("info string Thinking...")
        move, move_probs = self.mcts.get_move(self.board)
        
        # Información de búsqueda
        print(f"info depth 1 score cp 0 nodes {self.mcts.num_simulations}")
        print(f"bestmove {move.uci()}")
    
    def setoption(self, args):
        """Configura opciones del engine"""
        if "name" in args:
            name_idx = args.index("name")
            value_idx = args.index("value") if "value" in args else None
            
            if value_idx:
                option_name = args[name_idx + 1]
                option_value = args[value_idx + 1]
                
                if option_name == "Simulations":
                    self.mcts.num_simulations = int(option_value)
                    print(f"info string Set simulations to {option_value}")
    
    def run(self):
        """Loop principal UCI"""
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                
                parts = line.split()
                command = parts[0]
                args = parts[1:]
                
                if command == "uci":
                    self.uci()
                elif command == "isready":
                    self.isready()
                elif command == "ucinewgame":
                    self.board = chess.Board()
                elif command == "position":
                    self.position(args)
                elif command == "go":
                    self.go(args)
                elif command == "setoption":
                    self.setoption(args)
                elif command == "quit":
                    break
                
            except Exception as e:
                print(f"info string Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bot de Ajedrez con Machine Learning')
    parser.add_argument('mode', choices=['train', 'play', 'evaluate', 'uci'], 
                        help='Modo de operación')
    
    # Argumentos de entrenamiento
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Número de iteraciones de entrenamiento')
    parser.add_argument('--games', type=int, default=50, 
                        help='Partidas por iteración')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Épocas de entrenamiento por iteración')
    parser.add_argument('--simulations', type=int, default=400, 
                        help='Simulaciones MCTS')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=50000, 
                        help='Tamaño del replay buffer')
    
    # Argumentos de modelo
    parser.add_argument('--model-size', choices=['small', 'standard', 'large'], 
                        default='small', help='Tamaño del modelo')
    parser.add_argument('--model-path', type=str, 
                        help='Ruta al modelo guardado')
    parser.add_argument('--resume', type=str, 
                        help='Reanudar entrenamiento desde checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'play':
        play_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'uci':
        engine = UCIEngine(
            model_path=args.model_path,
            model_size=args.model_size,
            simulations=args.simulations
        )
        engine.run()