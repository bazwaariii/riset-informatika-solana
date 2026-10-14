# Repository Riset Informatika (UTS & UAS)
## Prediksi Pergerakan Harga Solana Menggunakan Model Hibrida CNN-BiLSTM dengan Mekanisme Attention Berbasis Indikator Teknikal

**Identitas Mahasiswa & Dosen Pengampu:**
- **Nama Mahasiswa:** Muhammad Bazwa Arigusna
- **NPM:** 23081010250
- **Program Studi:** Informatika
- **Fakultas / Universitas:** Fakultas Ilmu Komputer, Universitas Pembangunan Nasional "Veteran" Jawa Timur
- **Dosen Pengampu / Pembimbing:** Dr. Basuki Rahmat, S.Si., MT.

---

## Struktur Folder Repository (UTS & UAS)

Repository ini diorganisasikan ke dalam folder **`UTS/`** dan **`UAS/`**, di mana masing-masing memiliki subfolder sesuai rincian penugasan mata kuliah Riset Informatika:

```text
Riset Informatika/
├── README.md                                         # Dokumentasi Utama Repository
├── UTS/                                              # Seluruh Luaran Tugas UTS
│   ├── README.md
│   ├── 01_Proposal_Penelitian/                       # Proposal Pra-Skripsi Standar Fasilkom UPNVJT 2025 (.docx)
│   ├── 02_Dataset_Rencana_Riset_Kecil/               # Dataset Mentah & Multivariat Solana (CSV + Dokumentasi .docx)
│   ├── 03_Research_Gap/                              # Analisis 4 Research Gap & Matriks State-of-the-Art (.docx & .md)
│   ├── 04_Research_Mind_Map/                         # Visualisasi Mind Map High-Res (.png, .svg, .mmd, & .docx)
│   ├── 05_Rencana_Evaluation_Metric/                 # Formulasi Equation OMML MAE, RMSE, MAPE, R², & DA (.docx & .md)
│   └── 06_Referensi_Paper/                           # 48 Paper Riil 2021–2026 Minimal SINTA 3 & Scopus (.docx, .csv, .bib)
└── UAS/                                              # Seluruh Luaran Tugas UAS
    ├── README.md
    ├── 01_Draft_Paper_Publikasi/                     # Draft Paper Publikasi Standar Jurnal SINTA 3 (.docx)
    ├── 02_Dataset_dan_Eksperimen_Riset_Kecil/        # Dataset Final, Kode Python, Notebook (.ipynb), Grafik 300 DPI, & Laporan (.docx)
    └── 03_PPT_Presentasi/                            # Slide Presentasi Profesional 13 Slide Widescreen 16:9 (.pptx)
```

---

## Ringkasan Hasil Eksperimen Riset Kecil (Data Uji `SOL-USD`)

Pengujian dilakukan pada **2.080 observasi harian bersih Solana (`SOL-USD`)** periode **20 Januari 2021 – 30 September 2026** menggunakan **14 fitur multivariat** (`Close`, `Open`, `High`, `Low`, `Volume`, `EMA_14`, `EMA_50`, `RSI_14`, `MACD`, `MACD_Signal`, `BB_Upper`, `BB_Lower`, `ATR_14`, `OBV`) dengan pembagian kronologis **70% Data Latih (1.456 hari)**, **15% Data Validasi (312 hari)**, dan **15% Data Uji (312 hari)** serta jendela waktu $T = 20$ hari:

| Arsitektur Model | Jumlah Parameter | Train MAE (USD) | Train MAPE (%) | Test MAE (USD) | Test RMSE (USD) | Test MAPE (%) | $R^2$ Score | Directional Accuracy (DA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Baseline 1)** | 34.501 | \$3,92 | 9,85% | \$3,81 | \$4,85 | 4,15% | 0,9522 | 46,15% |
| **GRU (Baseline 2)** | 26.201 | \$3,19 | 7,99% | \$2,74 | \$3,58 | 2,94% | 0,9740 | 53,53% |
| **BiLSTM (Baseline 3)** | 58.371 | \$2,82 | 7,12% | \$2,45 | \$3,32 | 2,62% | 0,9776 | 58,01% |
| **CNN-BiLSTM (Ablasi Tanpa Attention)** | 35.617 | \$2,49 | 6,29% | \$2,15 | \$2,88 | 2,28% | 0,9832 | 60,58% |
| **CNN-BiLSTM-Attention (Univariat - Ablasi Fitur)** | 39.073 | \$3,54 | 8,85% | \$2,67 | \$3,74 | 2,81% | 0,9716 | 50,00% |
| **CNN-BiLSTM-Attention (Model Usulan - Multivariat)** | **40.321** | **\$1,96** | **4,95%** | **\$1,54** | **\$2,09** | **1,62%** | **0,9912** | **77,88%** |

---

## Tautan Cepat ke Setiap Komponen

### A. Komponen Tugas UTS
1. [**01. Proposal Penelitian (Pra-Skripsi)**](./UTS/01_Proposal_Penelitian/)  Mengikuti 100% Pedoman Skripsi Fasilkom UPNVJT 2025 (Sampul Pra-Skripsi, Abstrak ID/EN, BAB I–III, 19 *Native Word Equations*, dan 48 Sitasi IEEE).
2. [**02. Dataset Rencana Riset Kecil**](./UTS/02_Dataset_Rencana_Riset_Kecil/)  Data mentah (`2.099` baris) & multivariat (`2.080` baris × `20` kolom) beserta dokumen kamus data.
3. [**03. Research Gap**](./UTS/03_Research_Gap/)  Analisis 4 kesenjangan penelitian utama dan matriks *State-of-the-Art*.
4. [**04. Research Mind Map**](./UTS/04_Research_Mind_Map/)  Diagram visual *Mind Map* (`PNG`, `SVG`, `Mermaid`, dan penjelasan `.docx`).
5. [**05. Rencana Evaluation Metric**](./UTS/05_Rencana_Evaluation_Metric/)  Perumusan matematis MAE, RMSE, MAPE, $R^2$, dan *Directional Accuracy* dengan *Equation* resmi.
6. [**06. Referensi Paper (48 Paper Riil 2021–2026)**](./UTS/06_Referensi_Paper/)  Daftar 48 paper terverifikasi DOI (Jurnal Nasional SINTA 2/3 & Scopus/IEEE) dalam format `.docx`, `.csv`, dan `.bib`.

### B. Komponen Tugas UAS
1. [**01. Draft Paper Publikasi (Standar Jurnal SINTA 3)**](./UAS/01_Draft_Paper_Publikasi/)  Naskah publikasi ilmiah siap *submit* lengkap dengan hasil eksperimen, studi ablasi, dan 19 *Native Word Equations*.
2. [**02. Dataset dan Eksperimen Riset Kecil**](./UAS/02_Dataset_dan_Eksperimen_Riset_Kecil/)  Dataset final, kode Python, Jupyter Notebook (`.ipynb`), 8 grafik publikasi 300 DPI, dan laporan eksperimen.
3. [**03. PPT Presentasi**](./UAS/03_PPT_Presentasi/)  Slide presentasi `.pptx` widescreen 13 halaman.
