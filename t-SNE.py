import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from user_feature import per_feature
import nltk
nltk.download('averaged_perceptron_tagger')
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bert_embeddings(sentences, tokenizer):
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=700)
    with torch.no_grad():
        personalized_feature = per_feature(text_vec=inputs, text_data=tokens, text_original_data=sentences).to(device)
        fc2 = nn.Linear(6, 200).to(device)
        linear_layer = nn.LSTM(200, 200, num_layers=2, bidirectional=True, batch_first=True).to(device)
        personalized_feature = fc2(personalized_feature).to(device)
        personalized_feature = personalized_feature.unsqueeze(1).repeat(1, 200, 1).to(device)
        personalized_feature, _ = linear_layer(personalized_feature)
        personalized_feature = personalized_feature[:, -1, :]
        personalized_feature = personalized_feature.cpu().detach().numpy()
    return personalized_feature


tokenizer = BertTokenizer.from_pretrained('./BERTpre/bert-base-uncased')
# model = BertModel.from_pretrained('./BERTpre/bert-base-uncased')
# model = torch.load('best_model.pt')


# dataset
name = "KimKardashian"

balance = "/unbalance_c/"
texts = []
counts = {}
data_name = "../../Dataset/twitter_20_top/" + name + balance
a = data_name + 'cover.txt'
b = data_name + 'adg.txt'

filepaths = [a, b]
for filepath in filepaths:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            texts.append(line.strip())

for filepath in filepaths:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.readlines()
        filename = filepath.split('/')[-1]
        counts[filename] = len(content)

n = counts['cover.txt']
print(counts, n)
embeddings_np1 = bert_embeddings(texts[:n], tokenizer)
embeddings_np2 = bert_embeddings(texts[n:], tokenizer)
# embeddings_np = embeddings_np.numpy()


tsne = TSNE(n_components=2, perplexity=60)
tsne1 = TSNE(n_components=2, perplexity=15)
embeddings_2d1 = tsne.fit_transform(embeddings_np1)
embeddings_2d2 = tsne1.fit_transform(embeddings_np2)
# print(embeddings_2d)
# embeddings_2d_first = embeddings_2d[:n]  # First n embeddings
# embeddings_2d_rest = embeddings_2d[n:n+280]
embeddings_2d_first = embeddings_2d1  # First n embeddings
embeddings_2d_rest = embeddings_2d2

df_first = pd.DataFrame(embeddings_2d_first, columns=['Column1', 'Column2'])
df_rest = pd.DataFrame(embeddings_2d_rest, columns=['Column3', 'Column4'])
df_combined = pd.concat([df_first, df_rest], axis=1)

filename = "embedding.xlsx"
df_combined.to_excel(filename, index=False, header=False, engine='openpyxl')
print(f"Embedding saved to {filename}")

