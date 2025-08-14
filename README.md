# Legal Embedders - Türkçe Hukuki Embedding Modeli

Türkçe hukuki metinler için özelleştirilmiş gömme (embedding) modeli eğitimi ve değerlendirme araçları. Bu proje, Türk hukuk sistemine özgü belgeler, kanunlar, yargı kararları ve hukuki metinler üzerinde çalışan modern doğal dil işleme tekniklerini kullanır.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Hazırlama](#veri-hazırlama)
- [Model Eğitimi](#model-eğitimi)
- [Değerlendirme](#değerlendirme)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## ✨ Özellikler

- **🏛️ Hukuki Metin Özelleştirmesi**: Türk hukuk sistemi terminolojisi ve yapısına özel optimize edilmiş
- **📚 Çoklu Veri Kaynağı Desteği**: Yargıtay kararları, kanunlar, mevzuat, tebliğler gibi farklı hukuki doküman türleri
- **🔄 Veri Sentetizasyonu**: Gemma-3 27B modeli kullanarak otomatik soru-cevap çifti üretimi
- **⚡ Multi-GPU Eğitim**: Distributed Data Parallel (DDP) ile hızlandırılmış eğitim süreci
- **🎯 İleri Seviye Loss Fonksiyonları**: Multiple Negatives Ranking Loss (InfoNCE) ve opsiyonel Triplet Loss
- **📊 Performans Değerlendirme**: Retrieval kalitesi için comprehensive benchmark araçları

## 🚀 Kurulum

### Gereksinimler

```bash
# Python 3.8 veya üzeri gereklidir
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install sentence-transformers vllm
pip install tqdm pyyaml
```

### Hızlı Başlangıç

```bash
git clone https://github.com/your-username/legal-embedders.git
cd legal-embedders
pip install -r requirements.txt
```

## 💡 Kullanım

### Temel Kullanım

```python
from sentence_transformers import SentenceTransformer

# Eğitilmiş modeli yükle
model = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')

# Hukuki sorgu ve belgeler
query = "İş kanunu çalışma saatleri nelerdir?"
documents = [
    "İş Kanunu'na göre haftalık çalışma süresi 45 saattir...",
    "Fazla mesai ücretleri normal ücretin %50 fazlası olarak ödenir..."
]

# Embedding hesapla
embeddings = model.encode([query] + documents, normalize_embeddings=True)
scores = embeddings[0] @ embeddings[1:].T

print(f"En ilgili belge skoru: {scores.max():.3f}")
```

## 📊 Veri Hazırlama

### Sentetik Veri Üretimi

Projede Gemma-3 27B modeli kullanılarak otomatik soru-cevap çiftleri üretilir:

```bash
# Yargıtay kararları için soru üretimi
CUDA_VISIBLE_DEVICES=0 python synth/query_gen.py \
    --start 0 --end 5 \
    --ds_name fikriokan/ygty \
    --out_ds_name ygty-processed-1 \
    --prompt mev_kanun_prompt
```

### Desteklenen Veri Türleri

- **Yargıtay Kararları**: Türk mahkeme kararları ve içtihatları
- **Mevzuat**: Kanunlar, yönetmelikler ve tüzükler  
- **Tebliğler**: Resmi kurumlardan çıkan hukuki duyurular
- **Anayasa Maddeleri**: Türkiye Cumhuriyeti Anayasası

## 🏋️ Model Eğitimi

### Multi-GPU Eğitim Yapılandırması

```bash
# Accelerate yapılandırması (bir kez çalıştırın)
accelerate config

# DDP ile model eğitimi
accelerate launch train/train_st_embeddings_ddp.py \
    --data_path train/dataset.csv \
    --model_name intfloat/multilingual-e5-base \
    --output_dir models/turkish-legal-e5 \
    --batch_size 64 \
    --epochs 3 \
    --lr 2e-5 \
    --max_query_len 96 \
    --max_passage_len 384 \
    --use_triplet_if_negs \
    --margin 0.25
```

### Eğitim Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `batch_size` | Batch boyutu | 64 |
| `lr` | Öğrenme oranı | 2e-5 |
| `epochs` | Epoch sayısı | 3 |
| `max_query_len` | Maksimum sorgu uzunluğu | 96 |
| `max_passage_len` | Maksimum pasaj uzunluğu | 384 |

## 📈 Değerlendirme

### Performans Testi

```bash
python eval/single_run.py
```

Bu script, model performansını çeşitli hukuki sorgular üzerinde test eder ve retrieval accuracy metriklerini sağlar.

### Benchmark Sonuçları

Model performansı aşağıdaki metriklerle değerlendirilir:

- **Precision@K**: Top-K sonuç içindeki doğru eşleşme oranı
- **Recall@K**: Tüm ilgili belgeler içindeki bulunma oranı  
- **MRR (Mean Reciprocal Rank)**: Ortalama karşılıklı sıralama
- **NDCG (Normalized Discounted Cumulative Gain)**: Normalize edilmiş kümülatif kazanç

## 🗂️ Proje Yapısı

```
legal-embedders/
├── README.md                    # Bu dosya
├── synth/                       # Sentetik veri üretimi
│   ├── query_gen.py            # Soru üretimi scripti
│   ├── prompts.yaml            # Prompt şablonları
│   └── qposneg_prep.py         # Pozitif/negatif örnek hazırlama
├── train/                       # Model eğitimi
│   ├── train_st_embeddings_ddp.py  # DDP eğitim scripti
│   ├── default_config.yaml     # Accelerate yapılandırması
│   └── tiny.csv                # Örnek veri seti
├── eval/                        # Model değerlendirme
│   └── single_run.py           # Performans testi
└── playground/                  # Deneysel çalışmalar
    └── prompt_engineering.py   # Prompt optimizasyonu
```

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak istiyorsanız:

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

### Geliştirme Ortamı

```bash
# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks kurulumu
pre-commit install

# Testleri çalıştır
pytest tests/
```

## 📄 Lisans

Bu proje Apache 2.0 Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

```
Apache License 2.0

Copyright (c) 2025 Legal Embedders Projesi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 📞 İletişim

- **Proje Sahibi**: [GitHub Profili]
- **Issues**: [GitHub Issues Sayfası]
- **Tartışmalar**: [GitHub Discussions]

## 🙏 Teşekkürler

Bu proje aşağıdaki open-source projeleri ve araştırmaları temel alır:

- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [E5 Embedding Models](https://github.com/microsoft/unilm/tree/master/e5)

---

⭐ **Bu projeyi faydalı bulduysanız yıldız vermeyi unutmayın!**
