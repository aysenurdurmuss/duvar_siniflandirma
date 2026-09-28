from fastapi import FastAPI, UploadFile, File, HTTPException  # FastAPI sınıfı + dosya yükleme + hata döndürme için
import uvicorn  # sunucuyu bu dosyanın içinden başlatabilmek için uvicorn'u içe aktar
import tensorflow as tf  # modeli yüklemek ve tahmin almak için tensorflow'u içe aktar
from PIL import Image  # yüklenen dosyayı bir görsele çevirmek için
import io  # yüklenen ham veriyi görsel olarak okuyabilmek için
import datetime  # tahminlerin zamanını kaydetmek için
import threading  # dosyaya güvenli yazmak için kilit (Lock) mekanizmasını kullanabilmek için
from fastapi.middleware.cors import CORSMiddleware  # CORS ayarlarını otomatik yöneten hazır araç
import numpy as np

app = FastAPI()  # API'mizi temsil eden ana nesneyi oluştur

app.add_middleware(  # FastAPI'ye "her istek/cevapta şu ek kontrolü de yap" diyen bir katman ekliyoruz
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Model ve sabitler katmanı
model = tf.keras.models.load_model(  # sunucu açılırken eğitilmiş modeli diskten bir kez yükle
    'duvar_modeli_mobilenet.keras',
    custom_objects={'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input}
)

sinif_isimleri = ['catlak', 'kirik', 'saglam']  # eğitimde görülen alfabetik sıra, tahmin sonucunu isme çevirmek için

MAKS_DOSYA_BOYUTU = 5 * 1024 * 1024  # izin verilen en büyük dosya boyutu: 5 MB (bayt cinsinden)
GUVEN_ESIGI = 0.60  # model bu değerin altında bir güvenle tahmin ederse sonucu "belirsiz" say


ESIK_MESAFE = 24  # bu değerden daha uzaksa, görsel duvarla ilgisiz sayılır
taban_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
referans_embeddingler = np.load('referans_embeddingler.npy')



print("Model başarıyla yüklendi!")  # konsolda modelin yüklendiğini görmek için kontrol amaçlı

kilit = threading.Lock()  

def gunluge_yaz(dosya_adi, tahmin_sonucu, olasiliklar): 
    zaman = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    with kilit:  
        with open("tahmin_gunlugu.txt", "a", encoding="utf-8") as f:  
            f.write(f"{zaman} | dosya: {dosya_adi} | tahmin: {tahmin_sonucu} | olasiliklar: {olasiliklar}\n")  


def duvar(gorsel_dizisi_ham):  
    on_islenmis = tf.keras.applications.mobilenet_v2.preprocess_input(gorsel_dizisi_ham)  # taban model için ayrı bir ön işleme gerekiyor
    parmak_izi = taban_model.predict(on_islenmis, verbose=0)[0]

    farklar = referans_embeddingler - parmak_izi
    mesafeler = np.sqrt(np.sum(farklar ** 2, axis=1))
    en_yakin = np.min(mesafeler)

    return en_yakin <= ESIK_MESAFE  




@app.get("/")  # kök adrese ("/") bir GET isteği gelirse aşağıdaki fonksiyonu çalıştır
def ana_sayfa():
    return {"mesaj": "Merhaba, sunucu çalışıyor!"}  # tarayıcıya bu JSON'u geri döndür


@app.get("/siniflar")  # "/siniflar" adresine GET isteği gelirse çalışsın
def sinif_listesi():
    return {"siniflar": sinif_isimleri}  # mevcut sınıf listesini JSON olarak döndür


@app.post("/predict")  # "/predict" adresine POST isteği gelirse çalışsın
async def tahmin_yap(dosya: UploadFile = File(...)):

    #Fotoğrafı al ve doğrula
    if dosya.content_type not in ["image/jpeg", "image/png"]:  # yalnızca JPG/PNG kabul et
        raise HTTPException(status_code=400, detail="Geçersiz dosya formatı. Lütfen JPG veya PNG formatında bir görsel yükleyin.")
    icerik = await dosya.read()  # yüklenen dosyanın ham baytlarını oku
    if len(icerik) > MAKS_DOSYA_BOYUTU:  # dosya, izin verilenden büyükse
        raise HTTPException(status_code=400, detail="Dosya çok büyük. Lütfen 5 MB'tan küçük bir görsel yükleyin.")

    #Fotoğrafı hazırla
    try:
        gorsel = Image.open(io.BytesIO(icerik)).convert("RGB")  # ham baytları görsele çevir, 3 renk kanalına zorla
    except Exception:
        raise HTTPException(status_code=400, detail="Dosya açılamadı, bozuk ya da desteklenmeyen bir görsel olabilir.")
    gorsel = gorsel.resize((224, 224))  # modelin beklediği boyuta getir
    gorsel_dizisi = tf.keras.utils.img_to_array(gorsel)  # görseli sayısal diziye çevir
    gorsel_dizisi = tf.expand_dims(gorsel_dizisi, 0)  # başına "batch" boyutu ekle


    #Duvarla ilgili mi kontrolü
    if not duvar(gorsel_dizisi):
        return {
            "tahmin": "Duvar Değil",
            "mesaj": "Bu görsel bir duvar fotoğrafına yeterince benzemiyor, lütfen bir duvar görseli yükleyin."
        }



    # Modele sor
    tahmin = model.predict(gorsel_dizisi)  # modelden 3 sınıf için olasılık tahmini al
    tahmin_indeksi = int(tf.argmax(tahmin[0]))  # en yüksek olasılıklı sınıfın indeksini bul
    en_yuksek_olasilik = float(tahmin[0][tahmin_indeksi])  # kazanan sınıfın olasılık değerini sayı olarak al


    #Cevabı gönder (güven kontrolüyle)
    if en_yuksek_olasilik < GUVEN_ESIGI:  # model yeterince emin değilse
        gunluge_yaz(dosya.filename, "belirsiz", tahmin[0].tolist())  # bu belirsiz sonucu da logla
        return {
            "tahmin": "belirsiz",  # net bir sınıf ismi yerine belirsizlik bildir
            "olasiliklar": tahmin[0].tolist(),
            "mesaj": "Model bu görselden yeterince emin olamadı, lütfen daha net bir fotoğraf deneyin."
        }

    gunluge_yaz(dosya.filename, sinif_isimleri[tahmin_indeksi], tahmin[0].tolist())  # başarılı tahmini logla
    return {
        "tahmin": sinif_isimleri[tahmin_indeksi],  # tahmin edilen sınıf ismi
        "olasiliklar": tahmin[0].tolist()  # üç sınıfın olasılık değerleri
    }

#Başlatma katmanı
if __name__ == "__main__":  # bu dosya doğrudan çalıştırıldığında çalışsın
    uvicorn.run(app, host="127.0.0.1", port=8000)  # sunucuyu kendi bilgisayarında, 8000 portunda başlat





    