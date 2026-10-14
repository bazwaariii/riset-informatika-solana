# Research Mind Map: Prediksi Harga Solana (CNN-BiLSTM-Attention)

**Nama Mahasiswa:** Muhammad Bazwa Arigusna (NPM 23081010250)  
**Dosen Pengampu:** Dr. Basuki Rahmat, S.Si., MT.  

---

## 1. Diagram Visual Mind Map (High-Resolution)
![Research Mind Map Solana](./research_mind_map_solana.png)

## 2. Diagram Alir Mermaid
```mermaid
flowchart LR
    Core["<b>PREDIKSI HARGA SOLANA (SOL-USD)</b><br/>Model Hibrida CNN-BiLSTM-Attention<br/>Berbasis Indikator Teknikal"]

    Core --> B1["<b>1. Fenomena & Masalah</b><br/>• Blockchain Solana (PoH + PoS)<br/>• Volatilitas ekstrem & non-stasioner<br/>• Kelemahan prediksi univariat"]
    Core --> B2["<b>2. Research Gap</b><br/>• Degradasi memori LSTM/GRU pada lonjakan<br/>• Tanpa ekstraksi lokal antar-indikator<br/>• Ketiadaan selektivitas waktu (Attention)"]
    Core --> B3["<b>3. Dataset Multivariat</b><br/>• 2.080 Hari SOL-USD (2021–2026)<br/>• 14 Fitur: OHLCV + EMA, RSI, MACD, BB, ATR, OBV<br/>• Split Kronologis: 70% Train, 15% Val, 15% Test"]
    Core --> B4["<b>4. Arsitektur Model Usulan</b><br/>• 1D-CNN (Kernel k=3, LeakyReLU)<br/>• Bidirectional LSTM (Forward & Backward)<br/>• Temporal Bahdanau Attention + Skip Connection"]
    Core --> B5["<b>5. Evaluasi & Komparasi</b><br/>• Benchmark: LSTM, GRU, BiLSTM, CNN-BiLSTM<br/>• Studi Ablasi: Univariat vs Multivariat<br/>• Metrik: MAE, RMSE, MAPE, R², Directional Accuracy"]
    Core --> B6["<b>6. Luaran & Kontribusi</b><br/>• Proposal Skripsi UPNVJT + 48 Referensi SINTA/Scopus<br/>• Draft Paper Jurnal SINTA 3 + Eksperimen Riil<br/>• Sistem pendukung keputusan kuantitatif kripto"]
```

## 3. Daftar File di Folder Ini
- [`research_mind_map_solana.png`](./research_mind_map_solana.png) — Gambar Mind Map resolusi tinggi (300 DPI).
- [`research_mind_map_solana.svg`](./research_mind_map_solana.svg) — Grafik vektor Mind Map (SVG).
- [`research_mind_map.mmd`](./research_mind_map.mmd) — Kode sumber diagram Mermaid.
- [`Penjelasan_Research_Mind_Map_Solana.docx`](./Penjelasan_Research_Mind_Map_Solana.docx) — Dokumen uraian lengkap keenam cabang *Research Mind Map*.
