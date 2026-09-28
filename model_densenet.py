import tensorflow as tf

# DenseNet121'i ImageNet ağırlıklarıyla, kendi son katmanı olmadan yükle
taban_model = tf.keras.applications.DenseNet121(
    weights='imagenet',        # rastgele değil, hazır/eğitilmiş ağırlıklarla getir
    include_top=False,         # orijinal 1000 sınıflık son katmanı dahil etme
    input_shape=(224, 224, 3)  # görsel boyutu: 224x224 piksel, 3 renk kanalı (RGB)
)

# Omurgayı dondur - eğitim sırasında bu ağırlıklar değişmeyecek
taban_model.trainable = False

# Sıralı model: her adım, bir öncekinin çıktısını alıp işler
model = tf.keras.Sequential([
    # 1) Piksel normalizasyonu - DenseNet121'in beklediği standarda göre ölçekler
    tf.keras.layers.Lambda(
        tf.keras.applications.densenet.preprocess_input,
        input_shape=(224, 224, 3)
    ),
    # 2) Dondurulmuş DenseNet121 (omurga) - görselden özellik çıkarır
    taban_model,
    # 3) Özellik haritasını düz bir vektöre indirger
    tf.keras.layers.GlobalAveragePooling2D(),
    # 4) Yeni, 3 çıkışlı sınıflandırma katmanı (baş)
    tf.keras.layers.Dense(3, activation='softmax')
])
