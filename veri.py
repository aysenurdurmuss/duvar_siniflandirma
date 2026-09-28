import tensorflow as tf
import os

# Fiziksel çekirdek sayısı kadar (4) sınırlar — tek bir işlemin (mesela bir matris
# çarpımının) içinde kaç thread kullanılacağını belirler, CPU yükünü asıl bu kontrol ediyor
tf.config.threading.set_intra_op_parallelism_threads(4)

# Birbirinden bağımsız işlemlerin aynı anda kaç thread'le çalışacağını sınırlar —
# eğitim döngümüz sıralı (bir adım bitmeden diğeri başlamıyor) olduğu için büyük bir
# sayıya gerek yok
tf.config.threading.set_inter_op_parallelism_threads(1)

egitim_veri_seti = tf.keras.utils.image_dataset_from_directory(
    "veriseti/train",      # hangi klasör taranacak
    image_size=(224, 224), # her görseli bu boyuta getir (ResNet50'nin beklediği boyut)
    batch_size=8           # kaçarlı gruplar halinde modele verilecek
)

dogrulama_veri_seti = tf.keras.utils.image_dataset_from_directory(
    "veriseti/val",
    image_size=(224, 224),
    batch_size=8,
    shuffle=False           # doğrulamada karıştırmaya gerek yok, PyTorch'takiyle aynı sebep
)

print(egitim_veri_seti.class_names)