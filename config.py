import torch

# Define the configuration of the Transformer
BATCH_SIZE = 64
LR = 3e-4
N_LAYERS = 6
SEQ_LENGTH = 256
D_MODEL = 384
NUM_HEAD = 6
P_DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5_000