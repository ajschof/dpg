import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import os
import multiprocessing

# Import FASTA sequences
def import_fasta_sequences(file_path):
    sequences = []
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences.append(str(record.seq))
    return sequences

# Preprocess the data
def tokenize_sequences(sequences):
    amino_acids = sorted(set("".join(sequences)))
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}
    tokenized_sequences = [[aa_to_idx[aa] for aa in seq] for seq in sequences]
    return tokenized_sequences, aa_to_idx, idx_to_aa

# Dataset
class PolypeptideDataset(Dataset):
    def __init__(self, tokenized_sequences, max_seq_len, num_amino_acids):
        self.tokenized_sequences = tokenized_sequences
        self.max_seq_len = max_seq_len
        self.num_amino_acids = num_amino_acids

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, index):
        sequence = self.tokenized_sequences[index]
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)

        return input_sequence, target_sequence


# Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

def collate_fn(batch):
    input_sequences, target_sequences = zip(*batch)
    input_lengths = [len(seq) for seq in input_sequences]
    target_lengths = [len(seq) for seq in target_sequences]

    padded_input_sequences = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    padded_target_sequences = pad_sequence(target_sequences, batch_first=True, padding_value=0)

    return padded_input_sequences, padded_target_sequences

def main():
    parser = argparse.ArgumentParser(description="Train a model on polypeptide sequences")
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the FASTA file containing the polypeptide sequences")
    
    args = parser.parse_args()
    file_path = args.fasta_path

    sequences = import_fasta_sequences(file_path)
    tokenized_sequences, aa_to_idx, idx_to_aa = tokenize_sequences(sequences)

    max_seq_len = max([len(seq) for seq in tokenized_sequences])

    num_amino_acids = len(aa_to_idx)
    dataset = PolypeptideDataset(tokenized_sequences, max_seq_len, num_amino_acids)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Change the batch size to 128
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    # Set the number of threads to the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)

    device = torch.device("cpu")

    input_size = len(aa_to_idx)
    hidden_size = 128
    output_size = len(aa_to_idx)

    model = SimpleRNN(input_size, hidden_size, output_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, aa_to_idx)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

def train(model, train_loader, criterion, optimizer, device, aa_to_idx):
    model.train()
    epoch_loss = 0

    for input_sequence, target_sequence in tqdm(train_loader, desc="Training", unit="batch", mininterval=0.1):
        input_sequence = input_sequence.to(device)
        target_sequence = target_sequence.to(device)
        
        # One-hot encoding
        input_sequence = nn.functional.one_hot(input_sequence, num_classes=len(aa_to_idx)).float()

        optimizer.zero_grad()
        hidden = None
        output, _ = model(input_sequence, hidden)
        loss = criterion(output.reshape(-1, output.size(-1)), target_sequence.reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

if __name__ == "__main__":
    os.system('cls||clear')
    main()
