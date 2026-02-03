import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import os
import pickle
from collections import deque
import random
from tqdm import tqdm
import multiprocessing as mp

class ChessDataset(Dataset):
    """Dataset para entrenar la red neuronal"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board_tensor, policy_target, value_target = self.data[idx]
        return board_tensor, policy_target, value_target

class SelfPlayWorker:
    """Worker para generar partidas de self-play en paralelo"""
    def __init__(self, neural_net, num_simulations=200):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        
    def play_game(self, temperature_threshold=15):
        """Juega una partida completa contra sí mismo"""
        from mtcs import MCTS, board_to_tensor
        
        board = chess.Board()
        game_history = []
        move_count = 0
        
        mcts = MCTS(self.neural_net, num_simulations=self.num_simulations)
        
        while not board.is_game_over():
            # Temperatura alta al inicio, baja después
            temperature = 1.0 if move_count < temperature_threshold else 0.1
            mcts.temperature = temperature
            
            # Obtener estado actual
            board_tensor = board_to_tensor(board)
            
            # Ejecutar MCTS
            moves, probs = mcts.search(board)
            
            # Crear vector de política completo
            policy_vector = np.zeros(4672, dtype=np.float32)
            for move, prob in zip(moves, probs):
                move_idx = move.from_square * 64 + move.to_square
                if move_idx < 4672:
                    policy_vector[move_idx] = prob
            
            # Guardar (estado, política)
            game_history.append((board_tensor, policy_vector, board.turn))
            
            # Hacer movimiento
            move_idx = np.random.choice(len(moves), p=probs)
            board.push(moves[move_idx])
            move_count += 1
        
        # Asignar valor final a cada posición
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0
        
        # Crear datos de entrenamiento
        training_data = []
        for board_tensor, policy, turn in game_history:
            # El valor es desde la perspectiva del jugador actual
            value = final_value if turn == chess.WHITE else -final_value
            training_data.append((board_tensor, policy, value))
        
        return training_data, result

class ChessTrainer:
    """Sistema completo de entrenamiento"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Optimizador
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=config.get('replay_buffer_size', 50000))
        
        # Métricas
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'games_played': 0
        }
        
    def self_play(self, num_games=100, num_workers=4):
        """Genera partidas mediante self-play paralelo"""
        print(f"\nGenerando {num_games} partidas con {num_workers} workers...")
        
        worker = SelfPlayWorker(self.model, self.config.get('mcts_simulations', 200))
        
        all_training_data = []
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
        
        # Generar partidas
        for i in tqdm(range(num_games), desc="Self-play"):
            training_data, result = worker.play_game()
            all_training_data.extend(training_data)
            results[result] += 1
            
            if (i + 1) % 10 == 0:
                print(f"\nPartidas {i+1}/{num_games} - "
                      f"Blancas: {results['1-0']} | "
                      f"Negras: {results['0-1']} | "
                      f"Tablas: {results['1/2-1/2']}")
        
        # Añadir al replay buffer
        self.replay_buffer.extend(all_training_data)
        self.training_history['games_played'] += num_games
        
        print(f"\nTotal de posiciones en replay buffer: {len(self.replay_buffer)}")
        return results
    
    def train_epoch(self, batch_size=256, epochs=10):
        """Entrena la red con datos del replay buffer"""
        if len(self.replay_buffer) < batch_size:
            print("No hay suficientes datos en el replay buffer")
            return
        
        # Crear dataset y dataloader
        dataset = ChessDataset(list(self.replay_buffer))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_policy_loss = 0
            total_value_loss = 0
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for board_tensors, policy_targets, value_targets in pbar:
                # Mover a device
                board_tensors = board_tensors.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device).unsqueeze(1)
                
                # Forward pass
                policy_pred, value_pred = self.model(board_tensors)
                
                # Calcular pérdidas
                policy_loss = -torch.mean(torch.sum(policy_targets * policy_pred, dim=1))
                value_loss = nn.MSELoss()(value_pred, value_targets)
                loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Actualizar métricas
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'policy_loss': f"{policy_loss.item():.4f}",
                    'value_loss': f"{value_loss.item():.4f}",
                    'total_loss': f"{loss.item():.4f}"
                })
            
            # Guardar métricas de la época
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_total_loss = total_loss / num_batches
            
            self.training_history['policy_loss'].append(avg_policy_loss)
            self.training_history['value_loss'].append(avg_value_loss)
            self.training_history['total_loss'].append(avg_total_loss)
            
            print(f"\nEpoch {epoch+1} - "
                  f"Policy Loss: {avg_policy_loss:.4f} | "
                  f"Value Loss: {avg_value_loss:.4f} | "
                  f"Total Loss: {avg_total_loss:.4f}")
    
    def save_checkpoint(self, filepath):
        """Guarda checkpoint del modelo"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'replay_buffer_size': len(self.replay_buffer),
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint guardado en {filepath}")
    
    def load_checkpoint(self, filepath):
        """Carga checkpoint del modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Checkpoint cargado desde {filepath}")
        print(f"Partidas jugadas: {self.training_history['games_played']}")
    
    def train_loop(self, iterations=100, games_per_iteration=50, epochs_per_iteration=10):
        """Loop principal de entrenamiento"""
        print("=" * 60)
        print("INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        print(f"Iteraciones: {iterations}")
        print(f"Partidas por iteración: {games_per_iteration}")
        print(f"Épocas por iteración: {epochs_per_iteration}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"ITERACIÓN {iteration + 1}/{iterations}")
            print(f"{'='*60}")
            
            # Self-play
            results = self.self_play(num_games=games_per_iteration)
            
            # Entrenamiento
            self.train_epoch(epochs=epochs_per_iteration)
            
            # Guardar checkpoint cada 10 iteraciones
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(f"models/checkpoint_iter_{iteration+1}.pt")
        
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        self.save_checkpoint("models/final_model.pt")

if __name__ == "__main__":
    from neural_network import create_model
    
    # Configuración
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'replay_buffer_size': 50000,
        'mcts_simulations': 200
    }
    
    # Crear modelo
    model = create_model('small')  # Usar 'small' para pruebas rápidas
    
    # Crear trainer
    trainer = ChessTrainer(model, config)
    
    # Entrenar (prueba pequeña)
    trainer.train_loop(iterations=5, games_per_iteration=10, epochs_per_iteration=3)