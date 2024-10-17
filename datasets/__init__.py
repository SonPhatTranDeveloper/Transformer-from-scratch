"""
Author: Son Phat Tran
This file contains the logic for creating the Dataset for training
"""
import torch
from torch.utils.data import Dataset

from utils.tokeniser import encode
from utils.io import read_text

from config import SEQ_LENGTH


class TextDataset(Dataset):
    def __init__(self, file_name: str, seq_length: int = SEQ_LENGTH) -> None:
        """
        Initialise the text data loader
        :param file_name: the name of the file
        :param seq_length: size of each chunk (T)
        """
        # Read in the text
        text = read_text(file_name)

        # Encode to tokens
        tokens = encode(text)

        # Spilt into input sequence and output sequence
        x = []
        y = []
        for i in range(len(tokens) - seq_length):
            x.append(tokens[i:i + seq_length])
            y.append(tokens[i+1:i + seq_length + 1])

        # Convert to tensor
        self.x = torch.tensor(x, dtype=torch.long, device='cpu')
        self.y = torch.tensor(y, dtype=torch.long, device='cpu')

    def __len__(self):
        """
        Get the length of the dataset
        :return: the length of the dataset
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    # Create dataset
    text_dataset = TextDataset("../rawdata/kinh_van_hoa.txt", seq_length=8)
    print(len(text_dataset))