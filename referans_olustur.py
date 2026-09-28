import tensorflow as tf
import os
import numpy as np

taban_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

klasorler = ["catlak", "kirik", "saglam"]
alt_setler = ["train", "val"]
embeddingler = []

for alt_set in alt_setler:
    for klasor in klasorler:
        klasor_yolu = f"veriseti/{alt_set}/{klasor}"
        for dosya_adi in os.listdir(klasor_yolu):
            gorsel_yolu = f"{klasor_yolu}/{dosya_adi}"
            gorsel = tf.keras.utils.load_img(gorsel_yolu, target_size=(224, 224))
            gorsel_dizisi = tf.keras.utils.img_to_array(gorsel)
            gorsel_dizisi = tf.expand_dims(gorsel_dizisi, 0)
            gorsel_dizisi = tf.keras.applications.mobilenet_v2.preprocess_input(gorsel_dizisi)

            parmak_izi = taban_model.predict(gorsel_dizisi, verbose=0)
            embeddingler.append(parmak_izi[0])

        print(f"{alt_set}/{klasor} klasörü tamamlandı.")

embeddingler = np.array(embeddingler)
np.save("referans_embeddingler.npy", embeddingler)
print("Kaydedildi, toplam görsel sayısı:", len(embeddingler))