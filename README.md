![Banner](public/banner.jpg)

# DaveMod: AI Asisten Moderasi & Deteksi Pesan Penipuan

**DaveMod** adalah AI berbasis _neural network_ yang dirancang khusus untuk memoderasi pesan teks secara otomatis. Dilatih menggunakan dataset komprehensif, DaveMod mampu mengidentifikasi berbagai bentuk manipulasi digital—mulai dari tawaran kerja palsu, investasi bodong, promosi judi online, hingga _phishing_. AI DaveMod dilatih dari nol dengan dataset berisi berbagai pesan untuk memastikan AI ini efisien sesuai dengan fungsinya yang spesifik.

Berjalan langsung di _browser_ menggunakan **ONNX Runtime Web**, DaveMod tidak memerlukan ketergantungan ke API eksternal (seperti Gemini atau Claude). Pendekatan ini membuat pemrosesan menjadi ultra-responsif, menjamin privasi data secara penuh, serta efisien dengan ukuran model yang sangat ringan (hanya **170–250 KB**).

Nama Tim: BASIS-64
Kode Tim: MIT-009
Anggota: Gabriel Ado Ramos Tukan

---

## 10 Kategori Klasifikasi Pesan

DaveMod mampu menganalisis dan mengklasifikasikan pesan teks ke dalam **10 kategori spesifik**:

1. **`judol`**
   Mendeteksi pesan atau teks yang berisi ajakan taruhan judi online, termasuk yang menyamarkan kata kunci menggunakan kombinasi angka atau karakter tidak standar (_leet speak_).

2. **`tawaran_kerja_palsu`**
   Mengenali rekrutmen fiktif yang biasanya meminta uang muka/transfer serta menggunakan kontak personal seperti nomor WhatsApp pribadi (`08XX-XXXX-XXXX`) atau email domain gratisan (`@gmail.com`).

3. **`tawaran_kerja_asli`**
   Menandai penawaran karir resmi dari entitas terverifikasi yang tidak menggunakan email pribadi maupun nomor kontak personal.

4. **`giveaway_palsu`**
   Mendeteksi iming-iming hadiah atau undian gratis palsu yang mengarah pada pengumpulan data pribadi atau modus penipuan.

5. **`pinjol`**
   Mengidentifikasi promosi atau penawaran pinjaman online ilegal yang agresif dan tidak terverifikasi.

6. **`investasi_bodong`**
   Mengenali janji keuntungan fantastis tanpa risiko, skema cepat kaya, serta promosi aset kripto atau NFT ilegal.

7. **`investasi_asli`**
   Mendeteksi pesan informasi instrumen keuangan resmi seperti Reksadana dengan imbal hasil logis, laporan keuangan, pembagian dividen, berita resmi BEI/IDX, IHSG, serta pasar modal terdaftar.

8. **`phising`**
   Mengidentifikasi pesan manipulatif yang memberikan informasi palsu dan berupaya mencuri data sensitif melalui tautan berbahaya.

9. **`penting`**
   Kategori khusus untuk notifikasi penting seperti kode OTP (_One-Time Password_), pengumuman layanan publik, atau informasi pemadaman listrik.

10. **`netral`**
    Kategori untuk pesan kasual atau percakapan santai harian yang terindikasi aman dan tidak mengandung unsur penipuan.

---

### ✨ Key Highlights

- **100% Client-Side:** Tanpa API AI eksternal dan menjaga privasi pengguna secara penuh karena hanya _client-side_ (_zero data leak_).
- **Specific Use Case:** Model AI DaveMod dilatih dari **nol** agar efisien untuk tugas yang spesifik seperti deteksi pesan tanpa harus bergantung ke API AI eksternal yang seringkali boros dan jumlah request nya terbatas.
- **Model Ultra-Lightweight:** Ukuran _binary_ model AI sangat efisien, hanya **170–250 KB**.
- **Deteksi Kontekstual:** Andal mengenali pola bahasa gaul (_slang_), manipulasi angka, dan teks terselubung secara real-time.

# Keunggulan DaveMod AI

- **100% Client-Side Processing**  
  Model AI berjalan langsung di dalam browser menggunakan ONNX Runtime Web. Teks pesan tidak pernah dikirim atau disimpan ke server eksternal, menjamin privasi data 100% aman.

- **Analisis Real-Time Instan**  
  Mampu memproses kalimat, pola bahasa gaul (_slang_), maupun struktur pesan penipuan modern dalam hitungan milidetik saat tombol analisis ditekan atau fitur auto-deteksi aktif.

- **Deteksi Berbasis Konteks**  
  Dilatih secara spesifik untuk mengenali pola manipulatif seperti tawaran kerja palsu, tautan _phishing_, _giveaway_ palsu, promosi judi online, pinjol, hingga investasi bodong.

- **Model Efisien & Ringan**  
  Model AI DaveMod memiliki ukuran file sangat kecil (hanya 170–250 KB), memungkinkan pemuatan cepat bahkan di koneksi internet seluler yang lambat.

- **Transparansi Probabilitas**  
  Tidak sekadar memberikan hasil "Penipuan" atau "Netral", AI juga menyajikan distribusi probabilitas visual dan persentase tingkat keyakinan (_confidence score_) dari 5 kategori terbesar secara jelas (dari total 10 kategori berbeda).

- **Panduan Edukatif (Learn & Act)**  
  Memberikan rekomendasi langkah konkret secara kontekstual untuk mengedukasi pengguna (_Learn_) sekaligus panduan tindakan pertolongan pertama (_Act_) saat mendeteksi bahaya.

# Desclaimer

- Model machine learning ini beroperasi berdasarkan probabilitas statistik dan tidak dijamin 100% sempurna. Pesan sah (normal) terkadang dapat terdeteksi sebagai ancaman (False Positive). Selalu gunakan pertimbangan mandiri sebelum mengambil tindakan atau mengabaikan sebuah pesan.
- Platform ini sama sekali tidak mendukung aktivitas perjudian online ataupun penipuan, nama ataupun link yang digunakan pada sample hanyalah dummy data untuk keperluan testing model AI.
- Algoritma rule-based yang mendeteksi apakah pesan menggunakan nomor kontak pribadi atau email @gmail.com masih digunakan sebagai guardrail untuk mencegah false negative pada output AI (misalnya tawaran_kerja_palsu menjadi tawaran_kerja_asli)

# NextJS App

Project ini dibuat dengan NextJS.

## Scripts

- `pnpm run dev` untuk menjalankan NextJS dev server.
- `pnpm run build` builds the NextJS output.

# AI Model (Python)

Ini adalah AI model dari DaveMod, sistem inti yang digunakan untuk deteksi pesan.

## Model AI (di folder model)

- `python train.py` untuk melatih AI.
- `python predict.py` untuk menguji output AI.
- `./convert.sh` untuk konversi tflite ke format onnx.

## File (di folder model)

- `train.py` untuk melatih AI.
- `predict.py` untuk menguji output AI.
- `dataset.json` dataset ayng digunakan untuk melatih AI dari nol.
- `text_moderator.tflite` model AI.
- `text_moderator.onnx` model AI yang akan digunakan di client-side web dengan ONNX Runtime.
- `convert.sh` untuk konversi tflite ke format onnx.
