import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(  # eğitilmiş modeli diskten yükle
    'duvar_modeli_densenet.keras',
    custom_objects={'preprocess_input': tf.keras.applications.densenet.preprocess_input}  # Lambda içindeki fonksiyonu Keras'a tanıt
)

gorsel = tf.keras.utils.load_img(
    "test_gorselleri/saglam.png",  # test etmek istediğin görselin yolu
    target_size=(224, 224)                       # modelin beklediği boyuta getir
)

gorsel_dizisi = tf.keras.utils.img_to_array(gorsel)  # görseli sayısal diziye çevir

gorsel_dizisi = tf.expand_dims(gorsel_dizisi, 0)  # başına "batch" boyutu ekle

tahmin = model.predict(gorsel_dizisi)  # modelden 3 sınıf için olasılık tahmini al

sinif_isimleri = ['catlak', 'kirik', 'saglam']  # eğitimde görülen alfabetik sıra

tahmin_indeksi = np.argmax(tahmin[0])  # en yüksek olasılıklı sınıfın indeksini bul

print("Olasılıklar:", tahmin[0])
print("Tahmin:", sinif_isimleri[tahmin_indeksi])