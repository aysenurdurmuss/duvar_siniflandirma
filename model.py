import tensorflow as tf

# ResNet50'yi ImageNet ağırlıklarıyla, kendi son katmanı olmadan yükle
taban_model = tf.keras.applications.ResNet50(
    weights='imagenet',        # rastgele değil, hazır/eğitilmiş ağırlıklarla getir
    include_top=False,         # "bu görsel 1000 ImageNet kategorisinden hangisi" kararını veren kısmı ifade eder. bu kısmı hiç indirme
    input_shape=(224, 224, 3)  # görsel boyutu: 224x224 piksel, 3 renk kanalı (RGB)
)

# Omurgayı dondur - eğitim sırasında bu ağırlıklar değişmeyecek(taban_model nesnesinin trainable (eğitilebilir mi) özelliğini False yapıyoruz.)
taban_model.trainable = False

# Sıralı model: her adım, bir öncekinin çıktısını alıp işler
model = tf.keras.Sequential([

    # 1) Piksel normalizasyonu - ResNet50'nin beklediği standarda göre ölçekler
    tf.keras.layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,  # hazır normalizasyon fonksiyonu
        input_shape=(224, 224, 3)                          # bu katmanın da girdi boyutunu bilmesi gerekiyor
    ),

    # 2) Dondurulmuş ResNet50 (omurga) - görselden özellik çıkarır
    taban_model,

    # 3) 7x7x2048'lik özellik haritasını 2048 sayılık düz bir vektöre indirger
    tf.keras.layers.GlobalAveragePooling2D(),

    # 4) Yeni, 3 çıkışlı sınıflandırma katmanı
    tf.keras.layers.Dense(
        3,                    # çıkış sayısı: sağlam/çatlak/kırık
        activation='softmax'  # çıkışları toplamı 1 olan olasılıklara çevirir
    )
])