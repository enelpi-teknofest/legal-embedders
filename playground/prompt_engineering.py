from datasets import load_dataset#, get_dataset_config_names
from openai import OpenAI

ds_name = "fikriokan/sonbahcem-cukarn-batch-final"
ds_name = "fikriokan/ygty"
ds_name = "fikriokan/sonbahcem-tblg-batch-1"
ds_name = "fikriokan/sonbahcem-krm-batch-1"
ds_name = "fikriokan/sonbahcem-tuz-batch-final"
ds_name = "fikriokan/bloglar"
ds_name = "fikriokan/dnsy-1"

ds = load_dataset(ds_name)
cfgs = list(ds.keys())

client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="EMPTY"
)

prompt = """
Aşağıdaki metinden varlıkları (entities) ve aralarındaki ilişkileri çıkar.
JSON formatında döndür:

{{
    "entities": [
        {{"name": "varlık_adı", "type": "kişi/yer/organizasyon/kavram/olay", "description": "kısa açıklama"}},
    ],
    "relationships": [
        {{"from": "varlık1", "to": "varlık2", "relation": "ilişki_türü", "description": "ilişki açıklaması"}}
    ]
}}

KURALLAR:
- Sadece önemli ve anlamlı varlıkları çıkar (maksimum 5 entity)
- Belirsiz veya genel terimler kullanma
- İlişkiler net ve anlamlı olmalı
- Türkçe karakter kullan

Metin: {}
"""

doc = ds[cfgs[0]]['text'][4]

completion = client.chat.completions.create(
    model="google/gemma-3-27b-it",
    messages=[
        {
            "role": "user",
            "content": prompt.format(doc)
        }
    ],
)

print("Docs:", doc)
print("Soru:", completion.choices[0].message.content)
