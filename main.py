from tfidf import tokenize, build_vocabulary, compute_tf, compute_idf, compute_tfidf, cosine_similarity

sentence_a = "kedi köpeği kovaladı"
sentence_b = "köpek kediyi kovaladı"

tokens_a = tokenize(sentence_a)
tokens_b = tokenize(sentence_b)

documents = [tokens_a, tokens_b]

vocabulary = build_vocabulary(documents)

tf_a = compute_tf(tokens_a, vocabulary)
tf_b = compute_tf(tokens_b, vocabulary)

idf = compute_idf(documents, vocabulary)

tfidf_a = compute_tfidf(tf_a, idf)
tfidf_b = compute_tfidf(tf_b, idf)

similarity = cosine_similarity(tfidf_a, tfidf_b)

print(f"Benzerlik skoru: {similarity}")