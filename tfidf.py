import numpy as np

def tokenize(text):
    words = text.lower().split()    #lower() ile cümlenin tüm harflerini küçültüyoruz, split() ile kelimelere
    return words

def build_vocabulary(documents):    #documents değişkeni liste içinde liste alıyor [[a,b],[c,d]] gibi 
    all_items = []
    for tokens in documents:        #for döngüsüyle liste içindeki listedeki elemanları teker teker ayırıyoruz
        all_items.extend(tokens)    #teker teker ayırdığımız kelimeleri extend() ile listeye ekliyoruz
    uniqe = set(all_items)          #set() ile listede her kelimeden sadece bir tane olmasını sağlıyoruz
    ordered = sorted(uniqe)         #sorted() ile kelime listesini alfabetik sıralıyoruz
    return ordered

def compute_tf(tokens, vocabulary): #tokens: fonksiyona gelen cümle, vocabulary: bütün kelimeleri içeren liste
    arr = []
    for word in vocabulary:         #vocabulary'de ne kadar kelime varsa o sayıda döngü çalışır
        cnt = tokens.count(word)    #tokens metninde, vocabulary listesindeki kelimeler sırasıyla count() ile kaç tane olduğu sayılır
        arr.append(cnt)             #bu sayı(metinde geçen kelimenin kaç tane olduğu) arr listesine(dizisine) append() ile eklenir
    nump_arr = np.array(arr)        #yapılacak vektörel işlemler için arr listesi np.array() ile bir numpy dizisine çevrilir
    return nump_arr

def compute_idf(documents, vocabulary):
    n_docs = len(documents)             #kaç farklı cümle olduğunun sayısını alır
    idf_values = []                     #oluşturacağımız değer listesi için boş liste tanımlıyoruz
    for word in vocabulary:             #bütün kelimelerin listesinden baştan itibaren kelime kelime inceleyeceğimiz for döngüsü oluşturuyoruz
        doc_count = 0                   #kelimenin kaç farklı cümlede geçtiğini gösterecek sayaç  
        for tokens in documents:        #girilen cümledeki kelimeleri teker teker incelemek için for döngüsü oluşturuyoruz
            if word in tokens:          #tüm kelimelerin listesinden seçtiğimiz kelime girilen cümlenin herhangi bir yerinde var mı diye kontrol eder, true/false döndürür
                doc_count += 1          #aynıysa sayaç bir artıyor
        idf = np.log(n_docs/doc_count)  #girilen cümle uzunluğuyla seçili olan kelimenin kaç kere geçtiğini oranlayıp doğal logaritmaya alır
        idf_values.append(idf)          #bulunan oranı listeye ekler
    return np.array(idf_values) 

def compute_tfidf(tf_vector, idf_vector):
    return tf_vector * idf_vector       #vektörlerin aynı konumda bulunan elemanları çarpılıyor

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)      #A · B
    norm_a = np.linalg.norm(vec_a)          #linalg numpy kütüphanesinin lineer cebir alt modülü, norm() bir vektörün uzunluğunu hesaplar
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)  #(A · B) / (|A| × |B|)  
    
