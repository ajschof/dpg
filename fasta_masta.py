"""

FastaMasta: A PyTorch implementation of a simple RNN for protein sequence generation.
Author: Alex Schofield

Description:

A simple RNN model that [should] learn to generate protein sequences from biologically relevant FASTA sequences.
Whether or not it does that is the purpose of this project. It's a simple POC project that aims to find new possible proteins, fold them (secondary/tertiary) and show off the results in a cool way.

It also serves as a way for me to learn Python, PyTorch, and the basics of machine learning.

"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import os
import time
import sys
from colorama import Fore, Style, just_fix_windows_console
import torch.multiprocessing
from torch.utils.checkpoint import checkpoint
import gc
from textwrap import dedent

torch.multiprocessing.set_sharing_strategy("file_system")

# Making some changes and cleaning up a bit...
just_fix_windows_console()
gc.collect()
torch.cuda.empty_cache()

# Import FASTA sequences
def import_fasta_sequences(file_path):
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            yield str(record.seq)


# Preprocessing data
def tokenize_sequences(sequences):
    print("> Tokenizing sequences...")
    amino_acids = sorted(set("".join(sequences)))
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}
    tokenized_sequences = [np.array([aa_to_idx[aa] for aa in seq]) for seq in sequences]
    return tokenized_sequences, aa_to_idx, idx_to_aa


# Dataset
class PolypeptideDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.tokenized_sequences = tokenized_sequences

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, index):
        sequence = self.tokenized_sequences[index]
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)
        return input_sequence, target_sequence


# Collate function
def collate_fn(batch):
    input_sequences, target_sequences = zip(*batch)
    padded_input_sequences = pad_sequence(
        input_sequences, batch_first=True, padding_value=0
    )
    padded_target_sequences = pad_sequence(
        target_sequences, batch_first=True, padding_value=0
    )

    return padded_input_sequences, padded_target_sequences


# Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        def rnn_checkpoint(x, hidden):
            return self.rnn(x, hidden)

        def fc_checkpoint(output):
            return self.fc(output)

        output, hidden = checkpoint(rnn_checkpoint, x, hidden)
        output = checkpoint(fc_checkpoint, output)
        return output, hidden


def train(model, train_loader, criterion, optimizer, device, aa_to_idx):
    model.train()
    epoch_loss = 0

    num_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (input_sequence, target_sequence) in enumerate(train_loader):
        input_sequence = input_sequence.to(device)
        target_sequence = target_sequence.to(device)

        # One-hot encoding
        input_sequence = nn.functional.one_hot(
            input_sequence, num_classes=len(aa_to_idx)
        ).float()
        input_sequence.requires_grad = True

        optimizer.zero_grad()
        hidden = None
        output, _ = model(input_sequence, hidden)
        loss = criterion(
            output.reshape(-1, output.size(-1)), target_sequence.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # Calculate and display percentage and ETA
        percentage_left = 100 * (batch_idx + 1) / num_batches
        sys.stdout.write(
            f"\r{Fore.GREEN}{Style.BRIGHT}Training... {percentage_left:.2f}{Style.RESET_ALL}"
        )
        sys.stdout.flush()

    return epoch_loss / len(train_loader)


def validate(model, val_loader, criterion, device, aa_to_idx):
    model.eval()
    epoch_loss = 0

    num_batches = len(val_loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (input_sequence, target_sequence) in enumerate(val_loader):
            input_sequence = input_sequence.to(device)
            target_sequence = target_sequence.to(device)

            # One-hot encoding
            input_sequence = nn.functional.one_hot(
                input_sequence, num_classes=len(aa_to_idx)
            ).float()
            input_sequence.requires_grad = False

            hidden = None
            output, _ = model(input_sequence, hidden)
            loss = criterion(
                output.reshape(-1, output.size(-1)), target_sequence.reshape(-1)
            )

            epoch_loss += loss.item()

            # Calculate and display spinner, percentage, and ETA
            percentage_left = 100 * (batch_idx + 1) / num_batches
            sys.stdout.write(
                f"\r{Fore.YELLOW}{Style.BRIGHT}Validating... {percentage_left:.2f}{Style.RESET_ALL}"
            )
            sys.stdout.flush()

    return epoch_loss / len(val_loader)


def calculate_accuracy(model, data_loader, device, aa_to_idx):
    model.eval()
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for input_sequence, target_sequence in data_loader:
            input_sequence = input_sequence.to(device)
            target_sequence = target_sequence.to(device)

            # One-hot encoding
            input_sequence = nn.functional.one_hot(
                input_sequence, num_classes=len(aa_to_idx)
            ).float()
            input_sequence.requires_grad = False

            hidden = None
            output, _ = model(input_sequence, hidden)
            predicted_sequence = torch.argmax(output, dim=2)

            # Calculate number of correct predictions
            correct = (predicted_sequence == target_sequence).sum().item()
            total_correct += correct

            # Calculate total number of predictions
            total_predictions += target_sequence.numel()

    return total_correct / total_predictions


def save_checkpoint(state, filepath):
    torch.save(state, filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Train a model on polypeptide sequences"
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        required=True,
        help="Path to the FASTA file containing the polypeptide sequences",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the trained model"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Set number of workers to use - default is 8",
    )
    parser.add_argument(
        "--enable_mps",
        type=int,
        required=False,
        help="Use metal performance shaders (MPS) - 0 is off & 1 is on",
    )
    parser.add_argument(
        "--epoch", type=int, required=True, help="How many epochs to do?"
    )

    args = parser.parse_args()
    file_path = args.fasta_path
    output_path = args.output_path
    num_workerz = args.num_workers
    mps_enable = args.enable_mps
    num_e = args.epoch

    sequences = list(import_fasta_sequences(file_path))
    tokenized_sequences, aa_to_idx, idx_to_aa = tokenize_sequences(sequences)

    dataset = PolypeptideDataset(tokenized_sequences)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Ran out of CUDA memory, so batch size will be reduced from 128 to 64 for now...
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workerz,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=num_workerz,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("> Using CUDA acceleration")
        print(torch.cuda.get_device_name(0))

    elif torch.backends.mps.is_available() and mps_enable == 1:
        device = torch.device("mps")
        print(dedent(
            """
        > MPS enabled
        
        However, I've observed not so great results when using Metal acceleration.
        It appears that it only works efficiently when using large batch sizes, which I cannot test as I am limited to 16G RAM.
        Results may vary.
        """
        ))

    else:
        device = torch.device("cpu")
        print("> No CUDA (or MPS specified to be disabled)")

    input_size = len(aa_to_idx)
    hidden_size = 128
    output_size = len(aa_to_idx)

    model = SimpleRNN(input_size, hidden_size, output_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model

    num_epochs = num_e

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, aa_to_idx)
        val_loss = validate(model, val_loader, criterion, device, aa_to_idx)
        train_accuracy = calculate_accuracy(model, train_loader, device, aa_to_idx)
        val_accuracy = calculate_accuracy(model, val_loader, device, aa_to_idx)
        print(
            f" | Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        # Save a checkpoint after each epoch
        checkpoint_path = f"{output_path}_epoch{epoch+1}.pth"
        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        save_checkpoint(checkpoint_state, checkpoint_path)

        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    os.system("cls||clear")
    main()