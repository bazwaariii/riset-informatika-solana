# Draft Paper Publikasi (Standar Jurnal SINTA 3): Prediksi Harga Solana (`CNN-BiLSTM-Attention`)

**Judul Paper:** *Prediksi Pergerakan Harga Solana Menggunakan Model Hibrida CNN-BiLSTM dengan Mekanisme Attention Berbasis Indikator Teknikal*  
**Penulis:** Muhammad Bazwa Arigusna $^1$, Basuki Rahmat $^2$  
**Afiliasi:** Program Studi Informatika, Fakultas Ilmu Komputer, Universitas Pembangunan Nasional "Veteran" Jawa Timur  

---

## 1. File Utama di Folder Ini
- [`Draft_Paper_Publikasi_SINTA3_Solana_Muhammad_Bazwa_Arigusna_23081010250.docx`](./Draft_Paper_Publikasi_SINTA3_Solana_Muhammad_Bazwa_Arigusna_23081010250.docx) — Naskah publikasi ilmiah lengkap siap *submit* ke Jurnal Nasional Terakreditasi SINTA 2/3, dilengkapi Abstrak dwibahasa (ID/EN), 19 *native Word Equation (OMML)* (masing-masing disertai paragraf penjelasan variabel di bawahnya), 8 gambar resolusi tinggi (300 DPI, masing-masing disertai paragraf penjelasan visual di bawahnya), 3 tabel analisis eksperimen & studi ablasi (masing-masing disertai paragraf penjelasan di bawahnya), serta 48 referensi IEEE Style (2021–2026) yang **terurut 100% berurutan `[1]` s.d. `[48]` sesuai kemunculan di dalam naskah**.

## 2. Sorotan Hasil Eksperimen Utama pada Paper
- **MAE Data Uji:** `$1,54 USD` (turun 59,58% dari LSTM standar `$3,81 USD`)
- **RMSE Data Uji:** `$2,09 USD` (turun 56,91% dari LSTM standar `$4,85 USD`)
- **MAPE Data Uji:** `1,62%` (Kategori *Sangat Akurat* $< 10\%$, mengungguli GRU `2,94%`, BiLSTM `2,62%`, dan CNN-BiLSTM `2,28%`)
- **Koefisien Determinasi ($R^2$):** `0,9912`
- **Directional Accuracy (DA):** `77,88%` (meningkat drastis dibanding konfigurasi univariat `50,00%`)
