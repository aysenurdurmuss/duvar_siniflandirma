# Duvar Hasarı Sınıflandırma - Backend (FastAPI)

Duvar yüzeylerine ait görselleri "sağlam", "çatlak" ve "kırık" sınıflarına ayıran bir görüntü sınıflandırma modelini web servisi olarak sunan FastAPI uygulaması. Bir staj projesi kapsamında geliştirilmiştir.

Bu projenin arayüz (frontend) kısmı için: [duvar_siniflandirma_frontend](https://github.com/aysenurdurmuss/duvar_siniflandirma_frontend)

## Özellikler

- Transfer öğrenme ile eğitilmiş bir görüntü sınıflandırma modeli (ResNet50, MobileNet ve DenseNet mimarileri karşılaştırılarak en uygun olan seçildi)
- FastAPI ile `/predict` uç noktası üzerinden görsel yükleyip tahmin alma
- Yalnızca JPG/PNG formatındaki ve belirli boyut altındaki dosyaları kabul eden girdi doğrulama katmanı
- Embedding tabanlı benzerlik kontrolü ile projeyle alakasız görsellerin (duvar olmayan görsellerin) reddedilmesi
- Swagger arayüzü (`/docs`) üzerinden servisin test edilebilmesi

## Kullanılan Teknolojiler

- Python
- FastAPI
- TensorFlow / Keras

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
uvicorn main:app --reload
```

Servis ayağa kalktıktan sonra `http://127.0.0.1:8000/docs` adresinden Swagger arayüzüne ulaşılabilir.

## Model Dosyaları Hakkında

Üç ayrı mimariyle eğitilmiş model dosyalarının (`.keras`) tamamı bu depoya dahil edilmiştir. Servis (`main.py`) çalışırken bunlardan yalnızca **MobileNetV2 tabanlı modeli** (`duvar_modeli_mobilenet.keras`) kullanır; en iyi sonucu bu mimari verdiği için tercih edilmiştir. ResNet50 ve DenseNet121 tabanlı modeller (`duvar_modeli.keras`, `duvar_modeli_densenet.keras`) mimari karşılaştırmasının bir parçası olarak depoda tutulmuştur ve `test.py` gibi dosyalarda referans amaçlı kullanılabilir. Servisi çalıştırmak için ek bir işlem yapmaya gerek yoktur. Modellerin nasıl eğitildiğine dair kod da yine bu depoda yer almaktadır.

## Veri Seti Hakkında

Model eğitiminde kullanılan ham görseller bu depoda yer almamaktadır. "Sağlam" ve "çatlak" sınıflarına ait görseller hazır bir veri setinden alınmış, "kırık" sınıfına ait görseller ise internetten toplanmıştır. Servisi test etmek için kullanılan örnek görseller de aynı sebeple (internetten alınmış olmaları) bu depoya dahil edilmemiştir; test etmek isteyenler kendi duvar/çatlak görsellerini kullanabilir.
