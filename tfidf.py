import numpy as np

def tokenize(text):
    words = text.lower().split()    #lower() ile metnin tüm harflerini küçültüyoruz, split() ile kelimelere
    return words

def build_vocabulary(documents):    #documents değişkeni liste içinde liste alıyor [[a,b],[c,d]] gibi 
    all_items = []
    for tokens in documents:        #for döngüsüyle liste içindeki listedeki elemanları teker teker ayırıyoruz
        all_items.extend(tokens)    #teker teker ayırdığımız kelimeleri extend() ile listeye ekliyoruz
    uniqe = set(all_items)          #set() ile listede her kelimeden sadece bir tane olmasını sağlıyoruz
    ordered = sorted(uniqe)         #sorted() ile kelime listesini alfabetik sıralıyoruz
    return ordered

def compute_tf(tokens, vocabulary): #tokens: fonksiyona gelen metin, vocabulary: bütün kelimeleri içeren liste
    arr = []
    for word in vocabulary:         #vocabulary'de ne kadar kelime varsa o sayıda döngü çalışır
        cnt = tokens.count(word)    #tokens metninde, vocabulary listesindeki kelimeler sırasıyla count() ile kaç tane olduğu sayılır
        arr.append(cnt)             #bu sayı(metinde geçen kelimenin kaç tane olduğu) arr listesine(dizisine) append() ile eklenir
    nump_arr = np.array(arr)        #yapılacak vektörel işlemler için arr listesi np.array() ile bir numpy dizisine çevrilir
    return nump_arr

def compute_idf(documents, vocabulary):
    """Vocabulary'deki her kelime için IDF değerini hesaplar."""
    pass

def compute_tfidf(tf_vector, idf_vector):
    """TF ve IDF vektörlerini çarparak TF-IDF vektörünü üretir."""
    pass

def cosine_similarity(vec_a, vec_b):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    pass
