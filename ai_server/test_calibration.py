import cv2
import numpy as np
import time

def main():
    print("📏 KALİBRASYON MODU BAŞLATILIYOR...")
    print("Bu modda robot hareket etmez. Sadece kamera koordinatlarını test edeceğiz.")
    print("Lütfen kameranın önüne renkli/belirgin bir obje koyun.")

    # Kamera Seçimi (RealSense genellikle 1'dir)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Ekran Merkezi
    center_x, center_y = 320, 240

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Görüntü İşleme (Siyah-Beyaz ve Threshold)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Eşik değeri (Ortam ışığına göre 80-150 arası değiştirilebilir)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obj_x, obj_y = center_x, center_y

        if contours:
            # En büyük objeyi bul
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                # Merkezini bul
                M = cv2.moments(c)
                if M["m00"] != 0:
                    obj_x = int(M["m10"] / M["m00"])
                    obj_y = int(M["m01"] / M["m00"])
                
                # Objeyi Çiz
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                cv2.circle(frame, (obj_x, obj_y), 5, (0, 0, 255), -1)

        # --- KALİBRASYON BİLGİSİ ---
        # Merkeze göre farklar
        diff_x = obj_x - center_x
        diff_y = obj_y - center_y
        
        # Robotun ne yapacağını tahmin et (Simülasyon)
        # BURASI ÖNEMLİ: Kamerandaki X robotun Y'si mi? Kamerandaki Y robotun X'i mi?
        # Genellikle: Cam X -> Robot Y (Ters), Cam Y -> Robot X (Ters)
        
        scale = 0.8 # 1 piksel kaç mm? (Bunu test edeceğiz)
        
        rob_move_x = -(diff_y * scale) # Kamera Y ekseni -> Robot X ekseni
        rob_move_y = -(diff_x * scale) # Kamera X ekseni -> Robot Y ekseni
        
        # Çizgiler (Merkezden objeye)
        cv2.line(frame, (center_x, center_y), (obj_x, obj_y), (255, 0, 0), 2)
        
        # Bilgileri Yaz
        text1 = f"Pixel Fark: X={diff_x}, Y={diff_y}"
        text2 = f"Robot Gitmeli: X={rob_move_x:.1f}mm, Y={rob_move_y:.1f}mm"
        
        cv2.putText(frame, "MERKEZ", (center_x-20, center_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Kalibrasyon Ekrani', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()