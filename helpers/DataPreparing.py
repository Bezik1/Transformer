import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def text_to_tensor(text, vocab, max_len):
    tensor = torch.tensor([vocab.get(word, vocab['<PAD>']) for word in text], dtype=torch.long)
    padded_tensor = torch.nn.functional.pad(tensor, pad=(0, max_len - len(text)), value=vocab['<PAD>'])
    return padded_tensor.unsqueeze(0)

def prepare_data_from_text(file_path, device):
    file = open(file_path, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()

    data = [line.strip().split('<EOS>') for line in lines]

    all_words = set(word for line in data for sentence in line for word in sentence.split())

    source_vocab = {word: idx + 3 for idx, word in enumerate(all_words)}
    target_vocab = {word: idx + 3 for idx, word in enumerate(all_words)}

    source_vocab['<PAD>'] = 0
    source_vocab['<SOS>'] = 1
    source_vocab['<EOS>'] = 2  # Add this line
    target_vocab['<PAD>'] = 0
    target_vocab['<SOS>'] = 1
    target_vocab['<EOS>'] = 2  # Add this line

    source_index_to_word = {idx+3: word for word, idx in source_vocab.items()}
    target_index_to_word = {idx+3: word for word, idx in target_vocab.items()}

    source_index_to_word[0] = '<PAD>'
    source_index_to_word[1] = '<SOS>'
    source_index_to_word[2] = '<EOS>'  # Add this line
    target_index_to_word[0] = '<PAD>'
    target_index_to_word[1] = '<SOS>'
    target_index_to_word[2] = '<EOS>'  # Add this line

    max_source_len = max(len(line[0].split()) for line in data)
    max_target_len = max(len(line[1].split()) for line in data)

    source_sentences = [text_to_tensor(line[0].split(), source_vocab, max_source_len).to(device) for line in data]
    target_sentences = [text_to_tensor(line[1].split(), target_vocab, max_target_len).to(device) for line in data]

    dataset = TensorDataset(torch.cat(source_sentences), torch.cat(target_sentences))

    batch_size = 64
    train_loader = DataLoader(dataset, batch_size, shuffle=True)

    return train_loader, source_vocab, target_vocab, source_index_to_word, target_index_to_word
