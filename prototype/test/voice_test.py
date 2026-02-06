import speech_recognition as sr
from faster_whisper import WhisperModel
import os
import time
import sys

# --- RENKLİ ÇIKTILAR İÇİN (TERMİNAL) ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class JetsonVoiceAssistant:
    def __init__(self):
        print(f"{Colors.HEADER}[SİSTEM] Başlatılıyor...{Colors.ENDC}")
        
        # 1. AYARLAR
        self.model_size = "small"
        self.device = "cuda" # Jetson için zorunlu
        self.compute_type = "float16" # Performans için
        self.current_lang = "tr" # Varsayılan dil
        self.mic_calibrated = False

        # 2. MODEL YÜKLEME (Sadece 1 kere yapılır)
        print(f"{Colors.BLUE}[MODEL] Faster-Whisper ({self.model_size}) GPU'ya yükleniyor...{Colors.ENDC}")
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print(f"{Colors.GREEN}[MODEL] Yüklendi! ✅{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}[HATA] Model yüklenemedi: {e}{Colors.ENDC}")
            sys.exit(1)

        # 3. MİKROFON AYARLARI
        self.recognizer = sr.Recognizer()
        # Başlangıçta enerji eşiğini manuel ayarla, kalibrasyonla güncellenir
        self.recognizer.energy_threshold = 300 
        self.recognizer.dynamic_energy_threshold = True

    def calibrate_noise(self):
        """Ortam gürültüsünü dinler ve filtreyi ayarlar"""
        try:
            with sr.Microphone() as source:
                print(f"\n{Colors.WARNING}🔇 Lütfen 1 saniye SESSİZ olun (Kalibrasyon)...{Colors.ENDC}")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self.mic_calibrated = True
                print(f"{Colors.GREEN}✅ Kalibrasyon Tamamlandı! (Eşik: {self.recognizer.energy_threshold}){Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}[HATA] Mikrofon hatası: {e}{Colors.ENDC}")

    def listen_and_transcribe(self):
        """Sesi kaydeder ve yazıya döker"""
        if not self.mic_calibrated:
            print(f"{Colors.WARNING}UYARI: Önce kalibrasyon yapmanız önerilir (Menüden 'c' seçin).{Colors.ENDC}")

        try:
            with sr.Microphone() as source:
                print(f"\n{Colors.BLUE}🎤 KONUŞUN! (Dinliyorum...){Colors.ENDC}")
                # timeout: Ses gelmezse kaç saniye beklesin
                # phrase_time_limit: Konuşma en fazla kaç saniye sürsün
                audio_data = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)
                
                print(f"{Colors.WARNING}⏳ İşleniyor...{Colors.ENDC}")
                
                # Geçici dosya oluştur
                temp_file = "temp_audio.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio_data.get_wav_data())
                
                # Transkripsiyon
                start_time = time.time()
                segments, info = self.model.transcribe(temp_file, language=self.current_lang)
                
                text = "".join([segment.text for segment in segments])
                duration = time.time() - start_time
                
                # Sonucu Göster
                print("-" * 40)
                print(f"{Colors.GREEN}🗣️  ALGILANAN ({self.current_lang}): {text}{Colors.ENDC}")
                print(f"{Colors.BLUE}⏱️  Süre: {duration:.2f} sn{Colors.ENDC}")
                print("-" * 40)

                # Temizlik
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except sr.WaitTimeoutError:
            print(f"{Colors.FAIL}❌ Ses algılanamadı (Zaman aşımı).{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}❌ Hata oluştu: {e}{Colors.ENDC}")

    def change_language(self):
        """Dil değiştirme menüsü"""
        lang = input(f"Dil kodu girin (tr, en, fr, de) [Şu an: {self.current_lang}]: ").strip()
        if lang:
            self.current_lang = lang
            print(f"{Colors.GREEN}✅ Dil değiştirildi: {self.current_lang}{Colors.ENDC}")

    def show_menu(self):
        """Ana Menü Döngüsü"""
        while True:
            print(f"\n{Colors.BOLD}--- JETSON SES KONTROL PANELİ ---{Colors.ENDC}")
            print("1. [r]  Kayıt Al ve Çevir (Record)")
            print("2. [c]  Gürültü Kalibrasyonu Yap (Calibrate)")
            print("3. [l]  Dili Değiştir (Language)")
            print("4. [q]  Çıkış (Quit)")
            
            choice = input(f"{Colors.BLUE}Seçiminiz: {Colors.ENDC}").lower().strip()

            if choice == '1' or choice == 'r':
                self.listen_and_transcribe()
            elif choice == '2' or choice == 'c':
                self.calibrate_noise()
            elif choice == '3' or choice == 'l':
                self.change_language()
            elif choice == '4' or choice == 'q':
                print("Çıkılıyor...")
                break
            else:
                print("Geçersiz seçenek, tekrar deneyin.")

if __name__ == "__main__":
    # Programı başlat
    assistant = JetsonVoiceAssistant()
    
    # Başlangıçta otomatik kalibrasyon yapalım
    assistant.calibrate_noise()
    
    # Menüyü göster
    assistant.show_menu()