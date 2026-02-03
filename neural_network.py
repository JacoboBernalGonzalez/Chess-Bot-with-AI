import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Bloque residual con conexiones skip"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    """Red Neuronal estilo AlphaZero para ajedrez"""
    def __init__(self, num_res_blocks=19, num_channels=256):
        super().__init__()
        
        # Entrada: 8x8x119 (representación del tablero)
        # 119 planos: 12 por pieza*posición, + historial de movimientos
        self.input_channels = 119
        
        # Convolución inicial
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Torre de bloques residuales
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head - predice probabilidad de cada movimiento
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)  # Todos los movimientos posibles
        
        # Value Head - evalúa la posición (-1 a +1)
        self.value_conv = nn.Conv2d(num_channels, 3, 1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # x shape: (batch_size, 119, 8, 8)
        
        # Torre convolucional
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 3 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_tensor):
        """Predicción para un solo tablero"""
        self.eval()
        with torch.no_grad():
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)
            policy, value = self.forward(board_tensor)
            return torch.exp(policy[0]), value[0].item()

class ChessNetSmall(nn.Module):
    """Versión más pequeña para entrenar más rápido (útil para prototipos)"""
    def __init__(self, num_res_blocks=10, num_channels=128):
        super().__init__()
        self.input_channels = 119
        
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(num_channels, 16, 1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 8 * 8, 4672)
        
        self.value_conv = nn.Conv2d(num_channels, 3, 1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 16 * 8 * 8)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 3 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_tensor):
        self.eval()
        with torch.no_grad():
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)
            policy, value = self.forward(board_tensor)
            return torch.exp(policy[0]), value[0].item()

def create_model(model_type='standard', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Factory para crear modelos"""
    if model_type == 'standard':
        model = ChessNet(num_res_blocks=19, num_channels=256)
    elif model_type == 'small':
        model = ChessNetSmall(num_res_blocks=10, num_channels=128)
    elif model_type == 'large':
        model = ChessNet(num_res_blocks=40, num_channels=384)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

if __name__ == "__main__":
    # Test de la red
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    model = create_model('standard', device)
    
    # Input de prueba
    dummy_input = torch.randn(4, 119, 8, 8).to(device)
    policy, value = model(dummy_input)
    
    print(f"Policy shape: {policy.shape}")  # (4, 4672)
    print(f"Value shape: {value.shape}")    # (4, 1)
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")