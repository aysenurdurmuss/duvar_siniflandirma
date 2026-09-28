import tensorflow as tf
import numpy as np

taban_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

referanslar = np.load("referans_embeddingler.npy")
print("Referans sayısı:", referanslar.shape)

def en_yakin_mesafe(gorsel_yolu):
    gorsel = tf.keras.utils.load_img(gorsel_yolu, target_size=(224, 224))
    gorsel_dizisi = tf.keras.utils.img_to_array(gorsel)
    gorsel_dizisi = tf.expand_dims(gorsel_dizisi, 0)
    gorsel_dizisi = tf.keras.applications.mobilenet_v2.preprocess_input(gorsel_dizisi)

    parmak_izi = taban_model.predict(gorsel_dizisi, verbose=0)[0]

    farklar = referanslar - parmak_izi
    mesafeler = np.sqrt(np.sum(farklar ** 2, axis=1))
    en_yakin = np.min(mesafeler)

    print(f"{gorsel_yolu}: en yakın mesafe = {en_yakin:.2f}")
    return en_yakin

en_yakin_mesafe("veriseti/val/catlak/00085.jpg")
en_yakin_mesafe("veriseti/val/catlak/00095.jpg")
en_yakin_mesafe("veriseti/val/kirik/image85.jpg")
en_yakin_mesafe("veriseti/val/kirik/image95.jpg")
en_yakin_mesafe("veriseti/val/saglam/00085.jpg")
en_yakin_mesafe("veriseti/val/saglam/00095.jpg")
en_yakin_mesafe("test_gorselleri/The_million_march_man.jpg")
en_yakin_mesafe("test_gorselleri/images.jpg")
en_yakin_mesafe("test_gorselleri/images1.jpg")
en_yakin_mesafe("test_gorselleri/images2.jpg")
en_yakin_mesafe("test_gorselleri/images8.jpg")
en_yakin_mesafe("test_gorselleri/images9.jpg")
en_yakin_mesafe("test_gorselleri/catlakdeneme.jpg")

