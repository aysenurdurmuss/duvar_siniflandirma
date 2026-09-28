import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

genel_model = tf.keras.applications.MobileNetV2(weights='imagenet')

def tahmin_goster(gorsel_yolu):
    gorsel = tf.keras.utils.load_img(gorsel_yolu, target_size=(224, 224))
    gorsel_dizisi = tf.keras.utils.img_to_array(gorsel)
    gorsel_dizisi = tf.expand_dims(gorsel_dizisi, 0)
    gorsel_dizisi = preprocess_input(gorsel_dizisi)

    tahmin = genel_model.predict(gorsel_dizisi)
    sonuclar = decode_predictions(tahmin, top=5)[0]

    print(f"\n{gorsel_yolu}:")
    for _, isim, olasilik in sonuclar:
        print(f"  {isim}: %{olasilik * 100:.1f}")

tahmin_goster("veriseti/train/catlak/00010.jpg")
tahmin_goster("veriseti/train/catlak/00035.jpg")
tahmin_goster("veriseti/train/catlak/00060.jpg")
tahmin_goster("veriseti/train/catlak/00075.jpg")
tahmin_goster("veriseti/train/kirik/image10.jpg")
tahmin_goster("veriseti/train/kirik/image35.jpg")
tahmin_goster("veriseti/train/kirik/image60.jpg")
tahmin_goster("veriseti/train/kirik/image75.jpg")
tahmin_goster("veriseti/train/saglam/00010.jpg")
tahmin_goster("veriseti/train/saglam/00035.jpg")
tahmin_goster("veriseti/train/saglam/00060.jpg")
tahmin_goster("veriseti/train/saglam/00075.jpg")
tahmin_goster("test_gorselleri/The_million_march_man.jpg")
tahmin_goster("test_gorselleri/images.jpg")
tahmin_goster("test_gorselleri/images1.jpg")
tahmin_goster("test_gorselleri/images2.jpg")
tahmin_goster("test_gorselleri/images8.jpg")
tahmin_goster("test_gorselleri/images9.jpg")