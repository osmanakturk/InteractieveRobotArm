import pyrealsense2 as rs
import numpy as np
import cv2

# --- KURULUM ---
pipeline = rs.pipeline()
config = rs.config()

# --- KRİTİK AYAR BURADA ---
# 15 cm altını görmek için çözünürlüğü düşürmek ZORUNDAYIZ.
# 424x240 çözünürlüğü, D435 kameralarda MinZ (Min Mesafe) değerini ~8 cm'ye çeker.
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)





# Akışı başlat
profile = pipeline.start(config)

# --- İYİLEŞTİRME AYARLARI ---
depth_sensor = profile.get_device().first_depth_sensor()

# High Density (Yüksek Yoğunluk) modunu açıyoruz
if depth_sensor.supports(rs.option.visual_preset):
    depth_sensor.set_option(rs.option.visual_preset, 4) 

# Hizalama nesnesi
align = rs.align(rs.stream.color)

# Mouse değişkenleri
point = (0, 0)
pressed = False

def mouse_callback(event, x, y, flags, param):
    global point, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and pressed:
        point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pressed = False

cv2.namedWindow('Yakin Mesafe Olcum')
cv2.setMouseCallback('Yakin Mesafe Olcum', mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # Hizalama
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Görüntüyü büyütelim (424x240 çok küçük görünebilir)
        color_image = np.asanyarray(color_frame.get_data())
        
        # Görüntü küçük olduğu için ekranda rahat görmek adına 2 katına çıkarabiliriz
        # Ancak hesaplamayı orijinal koordinatlarda yapmalıyız.
        
        if pressed:
            x, y = point
            # Sınır kontrolü (424x240 sınırlarına göre)
            if 0 <= x < 424 and 0 <= y < 240:
                dist = depth_frame.get_distance(x, y)
                
                # 0.00 geliyorsa hala kör noktada veya yüzey çok parlaktır

                text = f"{dist:.3f} m"
                color = (0, 255, 0)

                
                cv2.circle(color_image, (x, y), 3, color, -1)
                # Yazıyı biraz yukarı yazalım
                cv2.putText(color_image, text, (x - 20, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Görüntü çok küçükse büyütüp göster (Opsiyonel)
        display_image = cv2.resize(color_image, (848, 480))
        
        # Mouse koordinatlarını büyütülmüş ekrana göre ayarlamak zor olacağı için
        # şimdilik direkt küçük halini gösteriyorum. 
        # (Mouse koordinat dönüşümüyle uğraşmamak için)
        cv2.imshow('Yakin Mesafe Olcum', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()