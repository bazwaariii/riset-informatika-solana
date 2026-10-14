# Analisis Research Gap dan Matriks State-of-the-Art

**Nama Mahasiswa:** Muhammad Bazwa Arigusna (NPM 23081010250)  
**Dosen Pengampu:** Dr. Basuki Rahmat, S.Si., MT.  

---

## 1. Empat Kesenjangan Penelitian Utama (*Research Gaps*)
1. **Keterbatasan Input Univariat pada Volatilitas Tinggi Solana:** Sebagian besar riset prediksi harga Solana terdahulu (`[16]`, `[18]`, `[20]`) hanya mengandalkan harga penutupan masa lalu tanpa mengintegrasikan indikator momentum (*RSI-14, MACD*), volatilitas (*Bollinger Bands, ATR-14*), dan tekanan volume (*OBV*).
2. **Kurangnya Ekstraksi Morfologi Lokal Antar-Indikator pada RNN Standar:** Model *LSTM*, *GRU*, dan *BiLSTM* murni (`[12]`, `[13]`, `[14]`) memproses deret waktu secara langsung tanpa penyaringan konvolusional (*1D-CNN*) untuk menangkap interaksi lokal antar-indikator teknikal.
3. **Degradasi Memori pada Sekuen Panjang Tanpa *Attention Mechanism*:** Arsitektur *CNN-BiLSTM* standar (`[25]`, `[26]`) memadatkan seluruh informasi sekuensial ke *hidden state* langkah terakhir secara seragam. Penambahan *Temporal Bahdanau Attention* (`[27]`, `[28]`, `[29]`, `[30]`) memberikan bobot adaptif $\alpha_t$ pada langkah waktu historis yang paling krusial.
4. **Absennya Evaluasi Ketepatan Arah (*Directional Accuracy*) dan Studi Ablasi:** Riset terdahulu mayoritas hanya mengukur *point-error* (RMSE/MAPE) tanpa menguji ketepatan arah naik/turun harian (*Directional Accuracy*) serta kontribusi fitur multivariat terhadap univariat.

## 2. Matriks Perbandingan *State-of-the-Art* (2021–2026)

| Ref. | Penulis & Tahun | Objek Aset | Metode / Arsitektur | Fitur Input | Metrik Evaluasi | Keterbatasan / Celah Penelitian (*Gap*) |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| [1] | Rizkilloh & Widiyanesti (2022) [SINTA 2] | Cryptocurrency (BTC) | LSTM | OHLCV Harian | RMSE, MAPE | Hanya menguji arsitektur tunggal LSTM tanpa perbandingan adil dengan GRU/BiLSTM dan tanpa fitur momentum. |
| [2] | Seabe dkk. (2023) [Scopus Q1] | BTC, ETH, LTC | LSTM, GRU, BiLSTM | Harga Penutupan & Volume | RMSE, MAE, MAPE, R² | Representasi temporal bertumpu pada hidden state akhir tanpa pembobotan selektif Attention. |
| [11] | Hamayel & Owda (2021) [Scopus Q1] | BTC, ETH, LTC | LSTM, GRU, BiLSTM | Harga Historis Univariat | RMSE, MAE, MAPE | Belum menyertakan indikator teknikal, ekstraksi konvolusional 1D-CNN, maupun mekanisme Attention. |
| [12] | Priadinata dkk. (2025) [SINTA 2] | Cryptocurrency | LSTM, GRU, BiLSTM | Data Deret Waktu OHLC | RMSE, MAPE | Belum mengeksplorasi aset jaringan high-throughput Solana dan belum menerapkan ekstraksi fitur 1D-CNN. |
| [13] | Yanimaharta & Santoso (2025) [SINTA 3] | Cryptocurrency | LSTM, GRU, BiLSTM | Data Historis Kripto | RMSE, MAE, MAPE | Tidak mengintegrasikan indikator volatilitas (ATR, Bollinger Bands) dan belum menggunakan mekanisme Attention. |
| [16] | Aldo dkk. (2025) [IEEE] | Solana (SOL/USD) | Varian Arsitektur LSTM | Data Historis SOL/USD | RMSE, MAE, MAPE | Berfokus pada variasi lapisan LSTM murni tanpa ekstraksi fitur lokal 1D-CNN maupun indikator teknikal multivariat. |
| [17] | Pane dkk. (2025) [IEEE] | Solana (SOL/USD) | Stacked Ensemble | Fitur Lag & EMA | RMSE, MAE, MAPE | Hanya menggunakan fitur tren EMA dan lag tanpa osilator momentum (RSI, MACD) serta tanpa arsitektur sekuensial Attention. |
| [18] | Herdian (2026) [SINTA 3] | Solana (SOL/USD) | LSTM Framework | Data Harga Jangka Pendek | RMSE, MAE | Pendekatan univariat jangka pendek; rentan mengalami efek lagging saat volatilitas harian Solana melonjak. |
| [19] | Arya dkk. (2026) [SINTA 3] | Ethereum & Solana | LSTM Multi-Modal | Indikator Trading Dasar | RMSE, MAPE | Masih menggunakan backbone LSTM satu arah standar; belum memanfaatkan konteks dua arah BiLSTM dan Attention. |
| [24] | Malik, Parlika, & Kartini (2026) [SINTA 3] | Bitcoin (BTC) | BiLSTM | Dataset On-Chain & Harga | RMSE, MAPE | Menerapkan BiLSTM pada Bitcoin, namun belum mengintegrasikan lapisan 1D-CNN dan Temporal Attention pada Solana. |
| [25] | Diash, Nathania, Idhom, & Trimono (2025) [SINTA 2] | Ethereum (ETH) | Hybrid CNN-BiLSTM | Data Historis Harga ETH | RMSE, MAE, MAPE | Sudah menggabungkan CNN dan BiLSTM dengan baik, tetapi belum dilengkapi Temporal Attention dan belum diuji pada Solana. |
| Usulan | Arigusna & Rahmat (2026) [Penelitian Ini] | Solana (SOL-USD) | Hibrida CNN-BiLSTM + Temporal Attention | 14 Fitur Multivariat (OHLCV + 9 Indikator Teknikal) | MAE, RMSE, MAPE, R², Directional Accuracy (DA) + Studi Ablasi | Menjawab seluruh gap: ekstraksi lokal 1D-CNN lintas 14 fitur, konteks dua arah BiLSTM, pembobotan adaptif Temporal Attention, dan evaluasi ketepatan arah (DA). |