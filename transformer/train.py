import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data import Field

from transformer.data import load_SST


class TransformerClassification(nn.Module):
    def __init__(self, vocab_size, emb_size=512, d_model=512, nhead=8, num_layers=6, num_classes=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        emb = self.input_embedding(inputs.transpose(1, 0))
        output = self.transformer_encoder(emb.permute((1, 0, 2)))
        x = self.fc(output)

        return x


def train():
    train, val, test, TEXT, LABEL = load_SST(root_dir="/home/bane/code/Attention/.data")

    data_loader = DataLoader(train, batch_size=8, collate_fn=lambda x: generate_batch(x, TEXT, LABEL))

    model = TransformerClassification(len(TEXT.vocab))
    # optimizer = Adam(lr=1e-4)

    for inputs, labels in data_loader:
        # Run forward
        outputs = model(inputs)

        print("testing...")


def generate_batch(batch, TEXT: Field, LABEL: Field):
    texts, labels = zip(*[(example.text, example.label) for example in batch])

    inputs = TEXT.process(list(texts))
    labels = LABEL.process(list(labels))

    return inputs, labels


if __name__ == '__main__':
    train()