# Dataset Rencana Riset Kecil: Data Historis & Indikator Teknikal Solana (SOL-USD)

**Nama Mahasiswa:** Muhammad Bazwa Arigusna (NPM 23081010250)  
**Dosen Pengampu:** Dr. Basuki Rahmat, S.Si., MT.  

---

## 1. Daftar File Dataset di Folder Ini
1. [`solana_historical_raw_dataset.csv`](./solana_historical_raw_dataset.csv) — Data mentah (*raw*) harian OHLCV Solana (`SOL-USD`) dari Yahoo Finance (**2099 baris**, periode `2021-01-01` s.d. `2026-09-30`).
2. [`solana_multivariate_technical_indicators.csv`](./solana_multivariate_technical_indicators.csv) — Dataset bersih hasil *feature engineering* (**2080 baris × 20 kolom**, periode `2021-01-20` s.d. `2026-09-30`).
3. [`Dokumentasi_Dataset_Rencana_Riset_Kecil_Solana.docx`](./Dokumentasi_Dataset_Rencana_Riset_Kecil_Solana.docx) — Dokumen penjelasan lengkap kamus data, rumus *Equation* indikator teknikal, dan statistik deskriptif.

## 2. Ringkasan Statistik Harga Penutupan (`Close` SOL-USD)
- **Jumlah Observasi Bersih:** 2080 hari perdagangan
- **Harga Terendah (*Min*):** USD 2.99
- **Harga Tertinggi (*Max*):** USD 261.87
- **Rata-rata (*Mean*):** USD 98.42
- **Median:** USD 91.46
- **Simpangan Baku (*Std. Dev.*):** USD 66.82

## 3. Skema Pembagian Data Kronologis (*Time-Series Split*)
- **Data Latih (*Training Set* — 70%):** 1.456 observasi (`2021-01-20` s.d. `2025-01-14`)
- **Data Validasi (*Validation Set* — 15%):** 312 observasi (`2025-01-15` s.d. `2025-11-22`)
- **Data Uji (*Testing Set* — 15%):** 312 observasi (`2025-11-23` s.d. `2026-09-30`)
- **Ukuran Jendela Sekuensial (*Lookback Window*):** $T = 20$ *time steps* (hari)
