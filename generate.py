"""
Author: Son Phat Tran
This file contains the code implementation for text generation using the trained model
"""
import torch
import torch.nn.functional as F

from config import SEQ_LENGTH, N_LAYERS, DEVICE, GENERATION_LENGTH
from models import LanguageModel
from utils.tokeniser import ENCODER


def generate(language_model: LanguageModel, prompt: str, max_size: int = 10_000, device = DEVICE) -> str:
    """
    Generate a sequence of text based on the initial prompt
    :param device: device used for generation
    :param max_size: the total size of the sequence
    :param language_model: the language model
    :param prompt: the starting prompt
    :return: a sequence of string
    """
    # First, tokenize the prompt
    # token has size (1, prompt_length)
    # Remove the final newline symbol
    tokens = ENCODER.encode(prompt)[:-1]
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)

    # Place the model into eval mode
    language_model.eval()

    # Begin generation
    for _ in range(max_size):
        # Grab the latest SEQ_LENGTH characters from the tokens
        inputs = tokens[:, -SEQ_LENGTH:]

        # Put through the model
        # outputs has size (batch_size = 1, seq_length, n_vocab)
        _, outputs = language_model(inputs)

        # Reshape the output to get (batch_size = 1, n_vocab)
        outputs = outputs[:, -1, :]

        # Calculate the softmax to get the distribution
        probs = F.softmax(outputs, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Concatenate
        # Token now has size (batch_size, previous_length + 1)
        tokens = torch.cat((tokens, next_token), dim=1)

    # Convert all the generated tokens to list
    tokens = tokens[0].tolist()
    return ENCODER.decode(tokens)


if __name__ == "__main__":
    # Create and load the model
    model = LanguageModel(
        n_layer=N_LAYERS,
        n_vocab=len(ENCODER)
    )
    model.to(DEVICE)
    model = torch.compile(model)

    # Load the weights
    model.load_state_dict(
        torch.load("checkpoints/model_checkpoint_9.pth", map_location=DEVICE)
    )
    print("Model loaded!")

    # Generate text
    prompt = "Tiểu Long"
    text = generate(model, prompt, max_size=GENERATION_LENGTH)

    # Write to file
    with open('result.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    print("File written!")
