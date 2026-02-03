import math
import numpy as np
import chess
from collections import defaultdict

class MCTSNode:
    """Nodo del árbol de búsqueda Monte Carlo"""
    def __init__(self, board, parent=None, move=None, prior_prob=0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior_prob = prior_prob
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        """Valor promedio del nodo"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count, c_puct=1.5):
        """Upper Confidence Bound para selección"""
        # Exploration term
        u = c_puct * self.prior_prob * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        # Q-value (valor promedio)
        q = self.value()
        
        return q + u
    
    def select_child(self, c_puct=1.5):
        """Selecciona el mejor hijo según UCB"""
        return max(self.children.values(), 
                   key=lambda node: node.ucb_score(self.visit_count, c_puct))
    
    def expand(self, move_probs):
        """Expande el nodo creando hijos para todos los movimientos legales"""
        if self.is_expanded:
            return
        
        for move in self.board.legal_moves:
            move_idx = move_to_index(move)
            prior = move_probs[move_idx] if move_idx < len(move_probs) else 1e-8
            
            new_board = self.board.copy()
            new_board.push(move)
            
            child = MCTSNode(new_board, parent=self, move=move, prior_prob=prior)
            self.children[move] = child
        
        self.is_expanded = True
    
    def update(self, value):
        """Actualiza las estadísticas del nodo"""
        self.visit_count += 1
        self.value_sum += value
    
    def backpropagate(self, value):
        """Propaga el valor hacia arriba en el árbol"""
        self.update(value)
        if self.parent:
            # Cambiar perspectiva (el valor es desde el punto de vista del jugador actual)
            self.parent.backpropagate(-value)

class MCTS:
    """Monte Carlo Tree Search con red neuronal"""
    def __init__(self, neural_net, num_simulations=800, c_puct=1.5, temperature=1.0):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        
    def search(self, board):
        """Ejecuta búsqueda MCTS y retorna distribución de probabilidades"""
        root = MCTSNode(board)
        
        # Expandir nodo raíz
        board_tensor = board_to_tensor(board)
        policy, _ = self.neural_net.predict(board_tensor)
        root.expand(policy.cpu().numpy())
        
        # Ejecutar simulaciones
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selección: descender por el árbol hasta encontrar una hoja
            while not node.is_leaf() and node.is_expanded:
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Evaluación
            value = 0
            if node.board.is_game_over():
                # Posición terminal
                result = node.board.result()
                if result == "1-0":
                    value = 1 if node.board.turn == chess.WHITE else -1
                elif result == "0-1":
                    value = -1 if node.board.turn == chess.WHITE else 1
                else:
                    value = 0
            else:
                # Evaluar con la red neuronal
                board_tensor = board_to_tensor(node.board)
                policy, value = self.neural_net.predict(board_tensor)
                
                # Expandir nodo si no está expandido
                if not node.is_expanded:
                    node.expand(policy.cpu().numpy())
            
            # Backpropagation
            for node_in_path in reversed(search_path):
                node_in_path.update(value)
                value = -value  # Alternar perspectiva
        
        # Calcular distribución de movimientos
        return self._get_action_probs(root)
    
    def _get_action_probs(self, root):
        """Obtiene probabilidades de acción basadas en visit counts"""
        moves = []
        visit_counts = []
        
        for move, child in root.children.items():
            moves.append(move)
            visit_counts.append(child.visit_count)
        
        if self.temperature == 0:
            # Movimiento determinístico
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(visit_counts))
            probs[best_idx] = 1.0
        else:
            # Aplicar temperatura
            visit_counts = np.array(visit_counts, dtype=np.float32)
            visit_counts = visit_counts ** (1.0 / self.temperature)
            probs = visit_counts / visit_counts.sum()
        
        return moves, probs
    
    def get_move(self, board):
        """Retorna el mejor movimiento"""
        moves, probs = self.search(board)
        move_idx = np.random.choice(len(moves), p=probs)
        return moves[move_idx], dict(zip([m.uci() for m in moves], probs))

def move_to_index(move):
    """Convierte un movimiento chess.Move a índice"""
    # Simplificación: usar from_square * 64 + to_square
    # En una implementación completa, incluirías promociones
    return move.from_square * 64 + move.to_square

def board_to_tensor(board):
    """Convierte un tablero de ajedrez a tensor para la red neuronal"""
    import torch
    
    # Representación simplificada: 12 planos para piezas + extras
    tensor = np.zeros((119, 8, 8), dtype=np.float32)
    
    # Planos 0-11: piezas (6 blancas + 6 negras)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = square // 8
        file = square % 8
        piece_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[piece_idx, rank, file] = 1.0
    
    # Plano 12: turno (1 si es turno de blancas)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    
    # Planos 13-16: derechos de enroque
    tensor[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Plano 17: regla de 50 movimientos (normalizado)
    tensor[17, :, :] = board.halfmove_clock / 100.0
    
    # Planos 18+: historial de posiciones (simplificado)
    # En implementación completa: últimos 7 movimientos
    
    return torch.FloatTensor(tensor)

if __name__ == "__main__":
    # Test básico
    from neural_network import create_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model('small', device)
    
    mcts = MCTS(model, num_simulations=100)
    board = chess.Board()
    
    print("Ejecutando MCTS...")
    move, move_probs = mcts.get_move(board)
    print(f"Mejor movimiento: {move.uci()}")
    print(f"Top 5 movimientos:")
    for move_uci, prob in sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {move_uci}: {prob:.3f}")