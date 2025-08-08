from datasets import load_dataset#, get_dataset_config_names
from openai import OpenAI

ds_name = "fikriokan/sonbahcem-cukarn-batch-final"
ds_name = "fikriokan/ygty"
ds_name = "fikriokan/sonbahcem-tblg-batch-1"
ds_name = "fikriokan/sonbahcem-krm-batch-1"
ds_name = "fikriokan/sonbahcem-tuz-batch-final"
ds_name = "fikriokan/bloglar"

ds = load_dataset(ds_name)
cfgs = list(ds.keys())

client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="EMPTY"
)

prompt = """
Aşağıda sana bir hukuki metin vereceğim. Bu metin bir kanun, anayasa maddesi ya da Yargıtay kararına ait olabilir. 
Görevin, bu metni dikkatle analiz edip, içeriğiyle bağlantılı ama doğrudan atıf yapmayan, günlük bir insanın sorabileceği şekilde 
doğal ve genel bir hukuk sorusu oluşturmaktır.

Kurallar:
- Soru, verilen metinden türetilmiş bir temaya dayansın ama metindeki cümleleri doğrudan içermesin.
- Soru, genel hukuk bilgisini sorgulayan bir tarzda olsun.
- Soru, üniversite öğrencisi ya da hukuka ilgi duymayan biri tarafından sorulabilecek doğallıkta olsun.
- 1 veya 2 cümleden oluşabilir soru.
- Json formatında olup şöyle olması lazım: {\"genel_soru\":}

Şimdi metni vereceğim:
"""

doc = ds[cfgs[0]]['content'][4]

completion = client.chat.completions.create(
    model="google/gemma-3-27b-it",
    messages=[
        {
            "role": "user",
            "content": prompt + doc
        }
    ],
)

print("Docs:", doc)
print("Soru:", completion.choices[0].message.content)
