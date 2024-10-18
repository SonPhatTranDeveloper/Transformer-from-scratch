# Decoder-based Transformer from Scratch

This project implements a Decoder-based Transformer architecture from scratch using Python and PyTorch. The goal is to provide a clear, educational implementation of the Transformer model, focusing on the decoder component.

## Requirements

- PyTorch (model training).
- nltk (for tokenization).

## Installation

1) Clone the repo.
```
git clone https://github.com/SonPhatTranDeveloper/Transformer-from-scratch.git
cd Transformer-from-scratch
```
2) Install the library (please refer to https://pytorch.org/get-started/locally/ to install PyTorch locally).
```
pip install -r requirements.txt
```

## Model training

To train the model

1) Place the text corpus under ```rawdata/data.txt```.
2) Adjust the hyperparameters in the ```config.py``` file (depending on whether you are trying to train on GPU or CPU).
3) For text generation (during training to check if the model is training well), please replace the content in ```train.py``` file.
on code line ```172``` to your prompt.
```
prompt = "[YOUR PROMPT]"
```
4) Create a folder called ```checkpoints``` in the main directory to save model's weights.
4) Run the training process, using
```
python train.py
```

## Generate text

To generate text using trained model.

1) Replace the path to your trained model in ```generate.py``` on the code line ```70```
```
torch.load("checkpoints/[YOUR_MODEL_CHECKPOINT]", map_location=DEVICE)
```
2) Replace the prompt in ```generate.py``` on the code line ```75``` to your prompt
```
prompt = "[YOUR_PROMPT]"
```
3) Run the ```generate.py``` file
```
python generate.py
```

The generated content will be saved in the ```result.txt``` file in the root directory.

## Model details

- The model uses a simple word tokeniser to split text into tokens. You can try changing this to Byte-Pair Encoding for better results.
- The model roughly follows GPT architecture, but there are some differences.
- The implemented MultiHeadAttention is not an optimal implementation, we can replace it using batched Multi-head attention for faster processing (this will be done in future commits).
- There are some tricks used to accelerate the training process:
  - Mixed-precision floating numbers.
  - Compiling the model using ```torch.compile```.
