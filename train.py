import tensorflow as tf                                    # TensorFlow kütüphanesini içe aktar
from veri import egitim_veri_seti, dogrulama_veri_seti     # veri.py'den eğitim ve doğrulama setlerini al
from model import model                                    # model.py'den kurulmuş modeli al

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # ağırlıkları nasıl güncelleyeceğini belirler
    loss='sparse_categorical_crossentropy',                   # tahmin hatasını nasıl ölçeceğini belirler
    metrics=['accuracy']                                      # sadece raporlama için, doğruluk oranını da göster
)

erken_durdurma = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',              # hangi değeri takip edeceğini belirtir (doğrulama kaybı)
    patience=5,                      # kaç epoch boyunca iyileşme olmazsa duracağını belirtir
    restore_best_weights=True        # durunca en iyi sonucu veren ağırlıklara geri döner
)

gecmis = model.fit(
    egitim_veri_seti,                     # eğitimde kullanılacak veri seti
    validation_data=dogrulama_veri_seti,  # her epoch sonunda test edilecek veri seti
    epochs=30,                            # en fazla kaç tur eğitim yapılacağı
    callbacks=[erken_durdurma],           # erken durdurma kuralını devreye sok
    verbose=2                             # epoch başına tek satır özet göster
)

model.save('duvar_modeli.keras')  # eğitilmiş modelin tamamını (mimari + ağırlıklar) dosyaya kaydet