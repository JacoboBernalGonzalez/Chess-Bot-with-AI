## Bot de ajedrez tipo AlphaZero (PyTorch)

Este proyecto es un **motor de ajedrez con red neuronal y MCTS** al estilo AlphaZero:

- `neural_network.py`: define la red convolucional con *policy head* y *value head*.
- `mtcs.py`: implementación de **Monte Carlo Tree Search** (usa la red para evaluar).
- `trainer.py`: genera partidas de *self‑play* y entrena la red.
- `main.py`: modos de **entrenamiento** y **juego interactivo en consola**.
- `uci_engine.py`: ejecutable principal con CLI y modo **UCI** para usarlo en GUIs / lichess-bot.

---

## Requisitos

- Python 3.9+ recomendado.
- Paquetes Python (ver `requirements.txt`):
  - `torch`
  - `numpy`
  - `python-chess`
  - `tqdm`
  - `matplotlib` (para la visualización gráfica del entrenamiento)

Instalación de dependencias (desde la carpeta del proyecto):

```bash
pip install -r requirements.txt
```

---

## Estructura de archivos

- `neural_network.py`: modelos `ChessNet` y `ChessNetSmall`, y la factoría `create_model`.
- `mtcs.py`: clases `MCTS` y `MCTSNode`, y la función `board_to_tensor`.
- `trainer.py`: clase `ChessTrainer` y utilidades para *self‑play* y entrenamiento.
- `main.py`: funciones:
  - `train_mode(args)`: bucle de entrenamiento.
  - `play_mode(args)`: jugar humano vs bot en consola.
  - `evaluate_mode(args)`: esqueleto para evaluación contra otros engines.
- `uci_engine.py`: clase `UCIEngine` y **CLI principal** con los modos:
  - `train`
  - `play`
  - `evaluate`
  - `uci`
- `visualize_training.py`: script para **ver gráficamente las pérdidas de entrenamiento** a partir de un checkpoint.

La carpeta `models/` (creada automáticamente) guardará los checkpoints (`.pt`).

---

## Cómo entrenar el modelo

Desde la carpeta del proyecto (`bot ajedrez`), ejecuta:

```bash
python uci_engine.py train \
  --model-size small \
  --iterations 5 \
  --games 10 \
  --epochs 3 \
  --simulations 200 \
  --lr 0.001 \
  --buffer-size 50000
```

Esto hará:

- Jugar partidas de *self‑play* con `SelfPlayWorker` (en `trainer.py`).
- Rellenar un *replay buffer* con (tablero, política, valor).
- Entrenar la red con esos datos usando `ChessTrainer.train_loop`.
- Guardar un checkpoint cada 10 iteraciones y al final en `models/final_model.pt`.

Si ya tienes un checkpoint y quieres **reanudar**:

```bash
python uci_engine.py train \
  --model-size small \
  --simulations 200 \
  --lr 0.001 \
  --buffer-size 50000 \
  --resume models/final_model.pt
```

---

## Visualizar resultados de entrenamiento (gráfico)

Para ver cómo han ido evolucionando las pérdidas de entrenamiento (policy, value y total)
puedes usar el script `visualize_training.py`, que abre una ventana con un gráfico usando `matplotlib`.

Ejemplo:

```bash
python visualize_training.py --checkpoint models/final_model.pt
```

Esto:

- Carga el checkpoint indicado.
- Lee el diccionario `training_history` guardado por `ChessTrainer`.
- Muestra un gráfico con:
  - `Policy Loss`
  - `Value Loss`
  - `Total Loss`

Es una forma rápida y gráfica de ver si el entrenamiento está convergiendo o si necesitas ajustar
hiperparámetros (learning rate, número de simulaciones, tamaño del modelo, etc.).

---

## Cómo jugar contra el bot (consola)

Primero asegúrate de tener un modelo entrenado (por ejemplo `models/final_model.pt`).
Luego ejecuta:

```bash
python uci_engine.py play \
  --model-size small \
  --model-path models/final_model.pt \
  --simulations 400
```

Características:

- Tú juegas con **blancas**.
- Introduce movimientos en formato **UCI** (ejemplo: `e2e4`, `g8f6`).
- Escribe `exit` para salir.
- El bot usa MCTS (`mtcs.py`) con la red neuronal para elegir jugadas.

---

## Uso como motor UCI

Para usar el bot como motor UCI en una GUI (por ejemplo, Arena, CuteChess, lichess-bot):

1. En la GUI, añade un nuevo motor y selecciona el ejecutable:
   - Comando: `python`
   - Argumentos: `uci_engine.py uci --model-size small --model-path models/final_model.pt --simulations 400`
2. La GUI enviará comandos UCI (como `uci`, `isready`, `position`, `go`) y el motor responderá.

Desde terminal, puedes probarlo manualmente:

```bash
python uci_engine.py uci --model-size small --model-path models/final_model.pt
```

Luego escribe en stdin:

```text
uci
isready
position startpos
go
```

Y el engine devolverá un `bestmove`.

---

## Notas de implementación

- Los imports entre módulos ya están alineados:
  - Todo el código usa `mtcs.py` (`from mtcs import MCTS, board_to_tensor`) en lugar de un nombre incorrecto.
  - `uci_engine.py` reutiliza `train_mode`, `play_mode` y `evaluate_mode` desde `main.py` para los modos de la CLI.
- El espacio en el nombre de la carpeta (`bot ajedrez`) no afecta si ejecutas los comandos situándote dentro de la carpeta antes de llamar a `python`.
- La representación del tablero (`board_to_tensor`) y el espacio de acciones (4672 movimientos) están simplificados y pensados para prototipos/experimentos.

---

## Preparar para subir a GitHub

Recomendaciones:

- Mantener en el repositorio:
  - Código fuente (`*.py`)
  - `README.md`
  - `requirements.txt`
- Excluir:
  - Checkpoints pesados (`models/*.pt`)
  - Directorios de caché (`__pycache__/`)

El `.gitignore` incluido ya cubre estos casos (puedes ajustarlo a tus necesidades).

