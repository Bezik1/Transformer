import torch

def sentence_to_tensor(sentence, vocab, device):
    words = sentence.split()

    indices = [vocab.get(word, vocab['<PAD>']) for word in words]

    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    return tensor

def tensor_to_sentence(tensor, vocab):
    sentences = []
    for indices in tensor:
        words = [vocab[int(index)] for index in indices]
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences