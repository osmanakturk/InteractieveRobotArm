from ultralytics import YOLO

# Prompt-Free (İstemsiz) modelimizi yüklüyoruz
# Model dosyası yoksa otomatik olarak indirecektir
model = YOLO("models/yoloe-26m-seg-pf.pt")
#model = YOLO("models/yoloe-26m-seg.pt")
#model.set_classes(["pen"])

# M1 işlemcinin grafik gücünü kullanmak için device="mps" ekliyoruz
# Masaüstü fotoğrafınızın adını "masaustu.jpg" olarak ayarlayın
#results = model.predict(source=0, device="mps")
results = model.predict(source=0, show=True, verbose=False, conf=0.25)
# Sonucu ekranda göster ve kaydet
#results[0].show()
