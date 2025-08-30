import numpy as np
import wikipedia as wiki
import razdel
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


queries = ["Длинные вбрасывания мяча крикетистом вынудили англичан изменить футбольные правила.",
           "Литературоведы установили, что пушкинский биф был направлен на неправильного адресата.",
           "По мнению Джона Леннона, «Эй, Джуд!» можно интерпретировать как «Эй, Джон!»."]

# get content from articles
articles = ""
wiki.set_lang("ru")
articles += wiki.page("Ганн, Уильям (крикетист)").content + " "
articles += wiki.page("Ex ungue leonem (эпиграмма)").content + " "
articles += wiki.page("Hey Jude").content + " "

# number of relevant documents
K = 10

collection = [_.text for _ in list(razdel.sentenize(articles))]

#tokenization
q_tokens = []
for q in queries:
    q_tokens.append([_.text for _ in razdel.tokenize(q)])
tokenized = []
new_collection = []
for i in range(len(collection)):
    doc = collection[i]

    # delete section headers
    if doc[0] == '=':
        collection[i] = doc.split('\n')[1]

    doc = [_.text for _ in razdel.tokenize(collection[i])]
    if doc != []:
        tokenized.append(doc)
        new_collection.append(collection[i])
# delete empty docs
collection = new_collection


def dcg(rel):
    k = len(rel)
    r = np.arange(1, k + 1)
    return np.sum(rel / np.log10(r + 1))

#idcg
idcg = np.zeros(3)
expert_marks = np.zeros((len(collection), 3), dtype=int)
file = open("docs.txt", "r")
file.readline()
i = 0
for line in file.readlines():
    elems = line.split(' ')
    for j in range(3):
        expert_marks[i][j] = int(elems[j + 1])
    i += 1
file.close()
for j in range(3):
    #print("Запрос: ", queries[j])
    ideal_order = np.argsort(expert_marks[:, j])[::-1]
    #print("Идеальная выдача:")
    #for k in range(K):
    #    print(f"{k + 1}. {collection[ideal_order[k]]}")
    #    print(f"Оценка: {expert_marks[ideal_order[k], j]}")
    idcg[j] = dcg(expert_marks[ideal_order[:K], j])
    #print(f"IDCG = {idcg[j]}")
    #print()

# rank docs by similarity between vectors
def vector_model(qs, docs):
    similarity = np.zeros((len(collection), len(queries)))
    ndcg = np.zeros(3)
    for j in range(len(queries)):
        q = qs[:, j]
        for i in range(len(collection)):
            doc = docs[:, i]
            doc_norm = np.linalg.norm(doc)
            q_norm = np.linalg.norm(q)
            # cosine measure
            if doc_norm != 0 and q_norm != 0:
                similarity[i, j] = np.sum(doc * q) / (doc_norm * q_norm)
        # document ranking
        sorted_args = np.argsort(similarity[:, j])[::-1]
        weights = similarity[sorted_args, j]
        print("Запрос: ", queries[j])
        print("Предложения:")
        for k in range(K):
            print(f"{k + 1}. {collection[sorted_args[k]]}")
            print(f"Вес: {np.round(weights[k], 4)}")
        ndcg[j] = dcg(expert_marks[sorted_args[:K], j]) / idcg[j]
        print(f"NDCG = {ndcg[j]}\n")
    print(f"Mean NDCG = {np.mean(ndcg)}")


# E5 model

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = ["query: " + q for q in queries] + ["passage: " + doc for doc in collection]
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach().numpy().transpose()

print("\nE5 model\n")
vector_model(embeddings[:, :len(queries)] , embeddings[:, len(queries):])
print()

# LaBSE model

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
sentences = queries + collection
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
embeddings = model_output.pooler_output.detach().numpy().transpose()

print("\nLaBSE model\n")
vector_model(embeddings[:, :len(queries)] , embeddings[:, len(queries):])
