"""
Author: Son Phat Tran
This file contains the training logic for the Transformers
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datasets import TextDataset

from utils.tokeniser import ENCODER

from config import BATCH_SIZE, N_LAYERS, DEVICE, EPOCHS, LR, SEQ_LENGTH

from models import LanguageModel


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
    tokens = ENCODER.encode(prompt)
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
    # Create dataset
    text_dataset = TextDataset("rawdata/kinh_van_hoa.txt")
    train_dataset, test_dataset = random_split(text_dataset, [0.9, 0.1])

    # Create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Create model
    model = LanguageModel(
        n_layer=N_LAYERS,
        n_vocab=len(ENCODER)
    )
    model.to(DEVICE)

    # Create optimiser and loss function
    optimiser = optim.Adam(lr=LR, params=model.parameters())

    # Training loop
    for epoch in range(EPOCHS):
        # Put the model into training mode
        model.train()

        # Set up average loss
        average_loss = 0.0
        total_batches = 0

        # Go through each epoch and calculate the loss
        for inputs, targets in train_dataloader:
            # Convert to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Zero-out the gradient
            optimiser.zero_grad()

            # Calculate the output
            loss, _ = model(inputs, targets)
            print(loss)

            # Back-propagation
            loss.backward()
            optimiser.step()

            # Add to result
            average_loss += loss.float()
            total_batches += 1

        # Calculate the average loss
        average_loss = average_loss / total_batches
        print(f"EPOCH {epoch} - AVG LOSS: {round(average_loss, 5)}")

        # Check for evaluation
        if epoch % 500 != 0:
            print("-" * 100)
            continue

        # Turn on eval mode
        model.eval()

        # Perform evaluation
        evaluation_loss = 0.0
        evaluation_batches = 0

        for inputs, targets in test_dataloader:
            # Convert to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Calculate the output
            loss, _ = model(inputs, targets)

            # Add to result
            evaluation_loss += loss.float()
            evaluation_batches += 1

        # Calculate the average loss
        evaluation_loss = evaluation_loss / evaluation_batches
        print(f"EPOCH {epoch} - TEST AVG LOSS: {round(evaluation_batches, 5)}")

        # Generate
        prompt = "Chương 1"
        generated_text = generate(model, prompt, max_size=40)
        print(f"GENERATED TEXT: {generated_text}")

        # Save model
        torch.save(model.state_dict(), f"checkpoints/model_checkpoint_{epoch}.pth")
        print("-" * 100)




