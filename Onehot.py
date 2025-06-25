import numpy as np
import pandas as pd
from Bio import SeqIO


class OneHotEncoderProtein:
    def __init__(self):
        """Initialize the One-Hot encoder"""
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_index = {aa: i for i, aa in enumerate(self.amino_acids)}

    def clean_sequence(self, sequence):
        """
        Replace uncommon amino acids with 'X'
        """
        return ''.join([aa if aa in self.aa_index else 'X'
                        for aa in sequence.upper()])

    def one_hot_encode(self, sequence):
        """
        Perform one-hot encoding on a single protein sequence.
        Returns: a flattened feature vector of shape (seq_len * 20)
        """
        seq_len = len(sequence)
        encoded = np.zeros((seq_len, len(self.amino_acids)))
        for i, aa in enumerate(sequence):
            if aa in self.aa_index:
                encoded[i, self.aa_index[aa]] = 1
        return encoded.flatten()

    def parse_txt_fasta(self, file_path):
        """
        Parse a TXT/FASTA file and return sequence IDs and cleaned sequences
        Returns: (list of sequence IDs, list of sequences)
        """
        seq_ids = []
        sequences = []
        current_id = ""
        current_seq = ""

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        seq_ids.append(current_id)
                        sequences.append(self.clean_sequence(current_seq))
                    current_id = line[1:]
                    current_seq = ""
                else:
                    current_seq += line

        # Add the last sequence
        if current_id:
            seq_ids.append(current_id)
            sequences.append(self.clean_sequence(current_seq))

        return seq_ids, sequences

    def process_file(self, txt_path):
        """
        Process an entire FASTA/TXT file into a one-hot feature matrix.
        Returns: (feature matrix as a NumPy array, sequence ID list)
        """
        seq_ids, sequences = self.parse_txt_fasta(txt_path)
        features = np.array([self.one_hot_encode(seq) for seq in sequences])
        return features, seq_ids
