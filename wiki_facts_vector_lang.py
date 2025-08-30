import numpy as np
import wikipedia as wiki
import razdel
import stanza

queries = ["Длинные вбрасывания мяча крикетистом вынудили англичан изменить футбольные правила.",
           "Литературоведы установили, что пушкинский биф был направлен на неправильного адресата.",
           "По мнению Джона Леннона, «Эй, Джуд!» можно интерпретировать как «Эй, Джон!»."]
# get content from articles
try:
    articles = ""
    wiki.set_lang("ru")
    articles += wiki.page("Ганн, Уильям (крикетист)").content + " "
    articles += wiki.page("Ex ungue leonem (эпиграмма)").content + " "
    articles += wiki.page("Hey Jude").content + " "
except:
    file = open("texts.txt", 'r')
    articles = file.read()
    file.close()

# number of relevant documents
K = 5

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

def get_lemmas(analyzer, docs, all_lemmas):
    result = []
    docs = analyzer(docs)
    for _, doc in enumerate(docs.sentences):
        lemmas = []
        for word in doc.words:
            # without stop-words
            if (word.text[0] not in "1234567890" and 
                word.upos not in ['PUNCT', 'NUM', 'INTJ', 'ADP', 'CCONJ', 'PART' 'SYM', 'X']):
                lemmas.append(word.lemma)
                all_lemmas.append(word.lemma)
        result.append(lemmas)
    return result, all_lemmas

nlp = stanza.Pipeline("ru", processors="tokenize,pos,lemma", 
                      tokenize_pretokenized=True, download_method=None)
# lemmatized queries
q_lemmas = get_lemmas(nlp, q_tokens, [])[0]
# lemmatized docs
lemmatized, all_lemmas = get_lemmas(nlp, tokenized, [])

all_lemmas = list(set(all_lemmas))
all_lemmas_dict = dict(zip(all_lemmas, list(range(len(all_lemmas)))))

# Vector space model

doc_tf = np.zeros((len(all_lemmas), len(collection)))
q_tf = np.zeros((len(all_lemmas), len(queries)))
df = np.zeros(len(all_lemmas))
for i in range(len(lemmatized)):
    doc = lemmatized[i]
    for lemma in doc:
        doc_tf[all_lemmas_dict[lemma]][i] += 1
    df[np.nonzero(doc_tf[:, i])] += 1
for i in range(len(q_lemmas)):
    q = q_lemmas[i]
    for lemma in q:
        if lemma in all_lemmas_dict:
            q_tf[all_lemmas_dict[lemma]][i] += 1

def vector_model(qs, docs):
    similarity = np.zeros((len(collection), len(queries)))
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
        print()

# tf
print("\nVector space model (tf)\n")
vector_model(q_tf, doc_tf)
#tfidf
N = doc_tf.shape[1]
idf = np.log10(N / df)
q_tfidf = q_tf * np.repeat(idf, q_tf.shape[1]).reshape(-1, q_tf.shape[1])
doc_tfidf = doc_tf * np.repeat(idf, doc_tf.shape[1]).reshape(-1, doc_tf.shape[1])
print("\nVector space model (tf-idf)\n")
vector_model(q_tfidf, doc_tfidf)

# Language model

def lang_model(lmbd):
    num_of_words = np.sum(doc_tf)
    q_probs = np.ones((len(collection), len(queries)))
    for j in range(len(queries)):
        word_idxs = np.nonzero(q_tf[:, j])
        # probability(word)
        word_probs = np.sum(doc_tf[word_idxs], axis=1) / num_of_words
        for i in range(len(collection)):
            doc_len = np.sum(doc_tf[:, i])
            # probability(word|document)
            doc_word_probs = doc_tf[word_idxs[0], i] / doc_len
            # probability(query|document)
            q_probs[i, j] = np.prod((1 - lmbd) * word_probs + lmbd * doc_word_probs)
        # document ranking
        sorted_args = np.argsort(q_probs[:, j])[::-1]
        weights = q_probs[sorted_args, j]
        print("Запрос: ", queries[j])
        print("Предложения:")
        for k in range(K):
            print(f"{k + 1}. {collection[sorted_args[k]]}")
            print(f"Вес: {weights[k]}")
        print()

# lambda = 0.5
print("\nLanguage model (lambda = 0.5)\n")
lang_model(0.5)
# lambda = 0.9
print("\nLanguage model (lambda = 0.9)\n")
lang_model(0.9)