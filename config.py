import torch

# Define the configuration of the Transformer
BATCH_SIZE = 32
LR = 3e-4
N_LAYERS = 6
SEQ_LENGTH = 256
D_MODEL = 384
NUM_HEAD = 6
P_DROPOUT = 0.2
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)
EPOCHS = 10
TEST_EPOCHS = 1
GENERATION_LENGTH = 10_000
