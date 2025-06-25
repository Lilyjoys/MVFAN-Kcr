import torch
import pandas as pd
from esm import pretrained
from Bio import SeqIO
import esm
from torch.utils.data import DataLoader, TensorDataset


def load_model(device):
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device)
    model.eval()
    return model, alphabet


def clean_sequence(sequence):
    return sequence.replace('O', 'X')


def parse_fasta(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    protein_name = ""
    sequence = ""
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if protein_name:
                data.append((protein_name, sequence))
            protein_name = line[1:]
            sequence = ""
        else:
            sequence += line
    if protein_name:
        data.append((protein_name, sequence))
    return data


def extract_features(fasta_file, model, alphabet, device, batch_size=8):
    sequences = parse_fasta(fasta_file)
    batch_converter = alphabet.get_batch_converter()
    batch_output = batch_converter(sequences)
    batch_labels, batch_strs, batch_tokens = batch_output
    batch_tokens = batch_tokens.to(device)

    dataset = TensorDataset(batch_tokens)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []
    for batch in data_loader:
        batch_token_sub = batch[0]
        with torch.no_grad():
            results = model(batch_token_sub, repr_layers=[12])
        # Use representation from layer 12 and average across residues
        embeddings = results["representations"][12]
        embeddings = embeddings.mean(dim=1)
        all_embeddings.append(embeddings)
        torch.cuda.empty_cache()

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def save_features_as_csv(embeddings, output_file):
    embeddings_np = embeddings.cpu().numpy()
    df = pd.DataFrame(embeddings_np)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")


def main():
    csv_files = [
        r"C:\Users\Admin002\PycharmProjects\pythonProject111\data\kcr\kcr_cv\kcr_cv+.txt",
        r"C:\Users\Admin002\PycharmProjects\pythonProject111\data\kcr\kcr_cv\kcr_cv-.txt",
        r"C:\Users\Admin002\PycharmProjects\pythonProject111\data\kcr\kcr_ind\Kcr_IND+.txt",
        r"C:\Users\Admin002\PycharmProjects\pythonProject111\data\kcr\kcr_ind\Kcr_IND-.txt"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, alphabet = load_model(device)

    for fasta_file in csv_files:
        file_name = fasta_file.split("\\")[-1].replace(".txt", "-esm2.csv")
        output_file = file_name
        embeddings = extract_features(fasta_file, model, alphabet, device, batch_size=4)
        save_features_as_csv(embeddings, output_file)


if __name__ == "__main__":
    main()
