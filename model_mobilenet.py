import tensorflow as tf

# MobileNetV2'yi ImageNet ağırlıklarıyla, kendi son katmanı olmadan yükle
taban_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',        # rastgele değil, hazır/eğitilmiş ağırlıklarla getir
    include_top=False,         # orijinal 1000 sınıflık son katmanı dahil etme
    input_shape=(224, 224, 3)  # görsel boyutu: 224x224 piksel, 3 renk kanalı (RGB)
)

# Omurgayı dondur - eğitim sırasında bu ağırlıklar değişmeyecek
taban_model.trainable = False

# Sıralı model: her adım, bir öncekinin çıktısını alıp işler
model = tf.keras.Sequential([
    # 1) Piksel normalizasyonu - MobileNetV2'nin beklediği standarda göre ölçekler
    tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        input_shape=(224, 224, 3)
    ),
    #  Dondurulmuş MobileNetV2 (omurga) - görselden özellik çıkarır
    taban_model,
    #  Özellik haritasını düz bir vektöre indirger
    tf.keras.layers.GlobalAveragePooling2D(),
    #  Yeni, 3 çıkışlı sınıflandırma katmanı 
    tf.keras.layers.Dense(3, activation='softmax')
])
