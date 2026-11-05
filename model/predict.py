import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load Metadata & Setup Tokenizer
print("Loading model and metadata...")
with open("moderation_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

categories = meta["categories"]
inv_label_map = {idx: cat for idx, cat in enumerate(categories)}
max_len = meta["max_len"]
vocab_size = meta["vocab_size"]
oov_token = meta["oov_token"]

# Rekonstruksi Tokenizer dari word_index yang disimpan saat training
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token, lower=True)
tokenizer.word_index = meta["word_index"]

# 2. Load TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path="text_moderator.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# def moderate_text(text):
#     # Preprocessing teks input
#     seq = tokenizer.texts_to_sequences([text])
#     padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
#     input_data = np.array(padded_seq, dtype=np.float32)
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
    
#     probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
#     predicted_class_id = np.argmax(probabilities)
#     confidence = probabilities[predicted_class_id]
#     predicted_label = inv_label_map[predicted_class_id]
    
#     print(f"Pesan    : '{text}'")
#     print(f"Prediksi : [{predicted_label.upper()}] (Confidence: {confidence*100:.2f}%)")
#     print("-" * 50)

def moderate_text(text):
    # Preprocessing teks input
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    input_data = np.array(padded_seq, dtype=np.float32)
    
    # PERBAIKAN: Duplikasi baris dari shape (1, max_len) menjadi (8, max_len)
    if input_data.shape[0] != 8 and False:
        input_data = np.repeat(input_data, 8, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Ambil hasil prediksi untuk data pertama (index [0])
    probabilities = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class_id = np.argmax(probabilities)
    confidence = probabilities[predicted_class_id]
    predicted_label = inv_label_map[predicted_class_id]
    
    print(f"Pesan    : '{text}'")
    print(f"Prediksi : [{predicted_label.upper()}] (Confidence: {confidence*100:.2f}%)")
    print("-" * 50)

# ==========================================
# PENGUJIAN PREDIKSI
# ==========================================
if __name__ == "__main__":
    print("\n=== PENGUJIAN MODEL MODERASI TEKS ===\n")
    
    test_messages = [
  "Halo, kami dari PT Maju Jaya mengundang Anda untuk mengikuti proses wawancara kerja besok pagi. Mohon siapkan CV Anda. Seluruh proses ini tidak dipungut biaya apa pun.",
  "Kerja frelance modal hp doang kak, dpt komisi 50rb per tgas. Cuma lke n subscribe yutub aja.",
  '''
  [PANGGILAN INTERVIEW & SELEKSI KARYAWAN PT ANGKASA JAYA TBK]
Berdasarkan evaluasi data resume yang Anda kirimkan melalui Jobstreet/Indeed, kualifikasi Anda memenuhi kriteria perusahaan kami.
Anda dijadwalkan hadir pada proses Wawancara & Seleksi pada:
🗓️ Hari/Tanggal: Senin, 2 November 2026
⏰ Waktu: 09.00 WIB - Selesai
📍 Lokasi: Gedung Antariksa Lt. 5, Jl. Sudirman No. 12, Jakarta Selatan.
KODE REGISTRASI: PT.AJ/049/2026
PENTING (WAJIB DIBACA):
Proses seleksi ini bekerja sama dengan agen travel resmi "Duta Tour & Travel" untuk akomodasi (tiket pesawat/hotel) bagi peserta dari luar kota. Biaya travel ditanggung sementara oleh peserta dan akan di-REIMBURSE (DIGANTI 100%) oleh perusahaan setelah wawancara selesai.
Silakan hubungi admin travel untuk reservasi tiket Anda:
WhatsApp Travel: 0812-3456-XXXX (Bpk. Rian)
Konfirmasi kehadiran Anda segera dengan membalas pesan ini: NAMA_HADIR/TIDAK HADIR.
  ''',
  '''
  Dari: recruitment@gotocompany.com
Kepada: ryan.hidayat@email.com
Subjek: Surat Penawaran Kerja (Job Offer) – UI/UX Designer – Ryan Hidayat
Kepada Yth.
Ryan Hidayat
Jakarta Selatan
Halo Ryan,
Terima kasih atas waktu dan diskusi yang sangat produktif selama rangkaian proses seleksi di perusahaan kami, mulai dari tahap seleksi portfolio, technical assignment, hingga wawancara akhir bersama tim Product Design.
Berdasarkan hasil evaluasi menyeluruh, dengan sukacita kami menginformasikan bahwa tim manajemen PT GoTo Gojek Tokopedia Tbk memutuskan untuk memberikan penawaran kerja kepada Anda untuk posisi UI/UX Designer (Level: Associate). Kami sangat terkesan dengan solusi desain yang Anda paparkan saat sesi studi kasus minggu lalu.
Sebagai ringkasan, berikut adalah detail poin penawaran yang kami ajukan:
• Status Kepegawaian: Karyawan Kontrak (Full-time Contract) untuk jangka waktu 12 bulan, dengan opsi pengangkatan menjadi karyawan tetap berdasarkan performa tahunan.
• Tanggal Mulai Bekerja: 1 Desember 2026.
• Skema Kerja: Hybrid (Minimal 3 hari kerja dari kantor/WFO di GoTo HQ Jakarta, dan 2 hari kerja dari rumah/WFH).
• Fasilitas Perusahaan: Penyediaan perangkat kerja MacBook Pro, asuransi kesehatan swasta Mandiri Inhealth, serta saldo tunjangan mobilitas bulanan via aplikasi Gojek.
Rincian mengenai nilai gaji pokok bulanan sebesar Rp12.500.000, Tunjangan Hari Raya (THR), serta benefit proteksi BPJS Ketenagakerjaan dan Kesehatan telah kami lampirkan secara hukum dalam dokumen Official_Offer_Letter_GoTo_Ryan.pdf yang ada di dalam lampiran email ini.
Mohon untuk meninjau seluruh dokumen yang terlampir. Jika Anda menerima penawaran ini, silakan berikan tanda tangan elektronik Anda pada dokumen tersebut dan kirimkan kembali sebagai balasan email ini paling lambat hari Jumat, 6 November 2026 pukul 17.00 WIB.
Jika ada poin dalam surat penawaran ini yang ingin Anda tanyakan atau negosiasikan lebih lanjut, silakan hubungi saya melalui email ini atau kontak resmi perusahaan di nomor (021) 2910-1072.
Kami sangat berharap Anda dapat bergabung dan tumbuh bersama untuk membawa dampak positif bagi jutaan pengguna layanan kami di Indonesia.
Selamat bergabung di keluarga besar GoTo!
Hormat kami,
Dian Sastrowardoyo
Lead Talent Acquisition
PT GoTo Gojek Tokopedia Tbk
Gedung Pasaraya Blok M, Lantai 6, Jl. Iskandarsyah II No. 2, Kebayoran Baru, Jakarta Selatan 12160
Situs Resmi: https://www.gotocompany.com/
  ''',
  '''
  Dari: hrd.ptgotogojektokopediatbk@gmail.com
Kepada: ryan.hidayat@email.com
Subjek: PANGGILAN INTERVIEW & SELEKSI KARYAWAN PT GO COMP COMPANY TBK
Kepada Yth.
Ryan Hidayat
Di Tempat
Selamat siang,
Berdasarkan hasil evaluasi data resume dan portofolio yang Anda kirimkan melalui platform Lowongan Kerja Nasional, dengan sukacita kami menginformasikan bahwa dokumen Anda memenuhi kriteria kualifikasi dari komite seleksi perusahaan kami untuk posisi UI/UX Designer.
Anda dinyatakan lulus seleksi administrasi awal dan diwajibkan untuk hadir mengikuti tahapan Wawancara Tatap Muka serta Pengarahan Kerja Akhir yang akan dilaksanakan pada:
• Hari / Tanggal: Senin, 2 November 2026.
• Waktu: Jam 09.00 WIB s/d Selesai.
• Lokasi Kantor: Gedung Pasaraya Blok M, Lantai 6, Jl. Iskandarsyah II No. 2, Kebayoran Baru, Jakarta Selatan.
• Fasilitas Peserta: Seluruh biaya akomodasi, tiket pesawat/kereta api, penginapan hotel, serta uang saku harian peserta akan ditanggung 100% oleh perusahaan melalui sistem penggantian (reimbursement) setelah proses wawancara selesai.
Mengingat kuota peserta yang sangat terbatas dan ketat, proses pengadaan transportasi dan reservasi akomodasi perjalanan wajib dilakukan secara kolektif melalui agen biro perjalanan resmi yang telah ditunjuk oleh perusahaan kami, yaitu "Nusantara Tour & Travel".
Peserta diwajibkan untuk segera melakukan pemesanan tiket dan kamar hotel kepada agen penanggung jawab kami sebelum hari Jumat, 30 Oktober 2026 pukul 17.00 WIB melalui kontak di bawah ini:
• WhatsApp Admin Travel: 0821-3456-7890 (Bpk. Aryo Perkasa)
• Format Konfirmasi: GOTO_RYAN HIDAYAT_HADIR
Tolong lampirkan bukti transfer pembayaran tiket Anda kepada admin travel tersebut agar nomor registrasi tes dan berkas pencairan klaim reimburse Anda di bagian keuangan kantor kami bisa segera diterbitkan. Keterlambatan melakukan reservasi tiket akan dianggap sebagai bentuk pengunduran diri dari proses seleksi ini.
Terima kasih atas kerja samanya. Kami tunggu kehadiran Anda di Jakarta.
Hormat kami,
Dian Sastrowardoyo
Head of Recruitment Team
PT GoTo Gojek Tokopedia Tbk
Gedung Pasaraya Blok M, Lantai 6, Jl. Iskandarsyah II No. 2, Kebayoran Baru, Jakarta Selatan 12160
  ''',
  '''
  "Halo, saya Rini dari Departemen HRD Shopee Indonesia. Kami sedang membuka lowongan kerja paruh waktu berbasis ponsel dari rumah.
Tugas: Anda hanya perlu memberikan Like pada video YouTube atau produk toko online yang kami kirimkan.
Gaji: Rp 200.000 - Rp 2.000.000 per hari. Gaji langsung ditransfer ke rekening Anda setelah tugas selesai.
Syarat: Hanya butuh HP, kuota internet, dan umur di atas 18 tahun.
Jika Anda tertarik untuk mengambil komisi pertama Anda, silakan balas pesan ini dengan mengetik "SAYA TERTARIK" untuk kami hubungi melalui Telegram."
  ''',
  
  '''
  PANGGILAN SELEKSI WAWANCARA KERJA PT. PERTAMINA (PERSERO)
Berdasarkan hasil seleksi berkas administrasi, Anda dinyatakan memenuhi syarat untuk mengikuti tahapan wawancara yang akan dilaksanakan pada:
• Hari/Tanggal: Senin, 2 November 2026
• Waktu: 09.00 WIB - Selesai
• Tempat: Gedung Utama Pertamina, Jakarta Pusat
PENTING: Untuk efisiensi biaya, akomodasi, dan transportasi (tiket pesawat & hotel), semua peserta wajib memesan tiket melalui agen travel resmi penunjukan kami: "BUANA TRAVEL" (Contact: 0812-0911-6769).
Seluruh biaya travel yang Anda keluarkan akan diganti secara penuh (reimbursement) oleh perusahaan setelah proses wawancara selesai. Kegagalan menghubungi agen travel di atas akan membatalkan status kelulusan Anda.
''',

'''
Subjek: Undangan Wawancara - Finance Advisor - PT Bank Central Asia Tbk
Yth. Rusdi Supriyanto,
Terima kasih atas ketertarikan Anda untuk bergabung dengan PT Bank Central Asia Tbk. Berdasarkan tinjauan kami terhadap resume Anda yang masuk melalui portal Jobstreet untuk posisi Management Trainee (MT), kami ingin mengundang Anda untuk mengikuti tahap wawancara awal.
Wawancara akan dilaksanakan secara daring (online) pada:
• Hari/Tanggal: Rabu, 4 November 2026
• Waktu: 10.00 - 11.00 WIB
• Platform: Zoom Meeting (Tautan akan dikirimkan setelah konfirmasi)
• Pewawancara: Tim Talent Acquisition BCA
Mohon konfirmasikan ketersediaan Anda dengan membalas email ini paling lambat Selasa, 3 November 2026 pukul 15.00 WIB dengan format: Nama_Hadir atau Nama_Reschedule.
PENTING: PT Bank Central Asia Tbk tidak memungut biaya apa pun dalam seluruh proses rekrutmen. Kami tidak pernah bekerja sama dengan agen perjalanan (travel) mana pun. Seluruh proses seleksi dilakukan secara transparan.
Hormat kami,
Tim Talent Acquisition
PT Bank Central Asia Tbk
www.bca.co.id
''',
'''
Halo Reza Auditore,
Salam kenal. Saya Perrel, Talent Scout dari Tokopedia.
Saya melihat profil Anda di LinkedIn dan sangat terkesan dengan pengalaman Anda selama 2 tahun terakhir di bidang Digital Marketing. Saat ini, tim kami sedang berkembang dan mencari seorang Senior Digital Marketing Specialist untuk bergabung di kantor Jakarta.
Melihat latar belakang Anda yang kuat dalam mengelola paid ads dan strategi SEO, saya rasa posisi ini akan sangat menarik untuk pertumbuhan karier Anda.
Jika Anda sedang terbuka untuk peluang baru, bolehkah saya meminta CV terbaru Anda? Kita juga bisa menjadwalkan obrolan santai via telepon selama 15 menit minggu ini untuk membahas posisi ini lebih detail.
Terima kasih atas waktunya, dan saya tunggu kabar baik dari Anda.
Salam hangat,
Perrel
Talent Acquisition Specialist | Tokopedia
'''

  ]
    
    for msg in test_messages:
        moderate_text(msg)
        
        
while(True):
    moderate_text(input("masukkan teks : "))