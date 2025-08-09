from sentence_transformers import SentenceTransformer

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Görev: Web arama sorgusuna uygun bilgiyi içeren pasajları getir
task = 'Given a Turkish search query, retrieve relevant passages written in Turkish that best answer the query'

queries = [
    get_detailed_instruct(task, 'Kolay bir kahvaltı tarifi nedir?'),
    get_detailed_instruct(task, 'Dış mekan yürüyüşü için en iyi saat hangisidir?')
]

documents = [
    "Güne enerjik başlamak için yulaf ezmesi, süt ve meyveyle hazırlanan basit bir kahvaltı hem pratik hem de besleyicidir. Üzerine biraz bal ve tarçın eklerseniz lezzeti artar.",
    "Sabah saatleri, özellikle 07:00 ile 10:00 arası, açık havada yürüyüş yapmak için idealdir. Bu saatlerde hava daha serin ve temiz olur, ayrıca gün ışığı vücut ritmini destekler.",
    "Türkiye'nin en uzun nehri Kızılırmak'tır. Sivas'tan doğar, Karadeniz'e dökülür ve yaklaşık 1.355 kilometre uzunluğundadır."
]

input_texts = queries + documents

model = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
scores = (embeddings[:2] @ embeddings[2:].T) * 100

for i, query in enumerate(queries):
    print(f"\nSorgu: {query.split('Query: ')[-1]}")
    for j, doc in enumerate(documents):
        print(f"   → Belge {j+1} Skoru: {scores[i][j]:.2f}")
        print(f"     İçerik: {doc[:80]}...")

"""
Sorgu: Kolay bir kahvaltı tarifi nedir?
   → Belge 1 Skoru: 67.36
     İçerik: Güne enerjik başlamak için yulaf ezmesi, süt ve meyveyle hazırlanan basit bir ka...
   → Belge 2 Skoru: 31.68
     İçerik: Sabah saatleri, özellikle 07:00 ile 10:00 arası, açık havada yürüyüş yapmak için...
   → Belge 3 Skoru: 7.06
     İçerik: Türkiye'nin en uzun nehri Kızılırmak'tır. Sivas'tan doğar, Karadeniz'e dökülür v...

Sorgu: Dış mekan yürüyüşü için en iyi saat hangisidir?
   → Belge 1 Skoru: 28.14
     İçerik: Güne enerjik başlamak için yulaf ezmesi, süt ve meyveyle hazırlanan basit bir ka...
   → Belge 2 Skoru: 78.02
     İçerik: Sabah saatleri, özellikle 07:00 ile 10:00 arası, açık havada yürüyüş yapmak için...
   → Belge 3 Skoru: 18.70
     İçerik: Türkiye'nin en uzun nehri Kızılırmak'tır. Sivas'tan doğar, Karadeniz'e dökülür v...
"""
