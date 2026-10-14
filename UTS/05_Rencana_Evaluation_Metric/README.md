# Rencana Evaluation Metric: Prediksi Harga Solana (CNN-BiLSTM-Attention)

**Nama Mahasiswa:** Muhammad Bazwa Arigusna (NPM 23081010250)  
**Dosen Pengampu:** Dr. Basuki Rahmat, S.Si., MT.  

---

## 1. Spesifikasi 5 Metrik Evaluasi Kuantitatif

Seluruh metrik dievaluasi pada skala harga asli dalam satuan **USD** setelah dilakukan *inverse Min-Max scaling* pada data uji (*testing set*):

### a. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

### b. Root Mean Square Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}$$

### c. Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### d. Koefisien Determinasi ($R^2$ Score)
$$R^2 = 1 - \frac{\sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}{\sum_{i=1}^{n} \left( y_i - \bar{y} \right)^2}$$

### e. Directional Accuracy (DA)
$$\text{DA} = \frac{100\%}{n} \sum_{i=1}^{n} \mathbb{I}\left( \text{sgn}(y_i - y_{i-1}) = \text{sgn}(\hat{y}_i - y_{i-1}) \right)$$

---

## 2. Kriteria Interpretasi Nilai MAPE
| Rentang Nilai MAPE | Kategori Akurasi Prediksi | Target Penelitian Ini |
| :---: | :--- | :---: |
| **$< 10\%$** | Sangat Akurat (*Highly Accurate*) | **Target Utama ($< 5\%$)** |
| **$10\% - 20\%$** | Baik (*Good Forecasting*) | — |
| **$20\% - 50\%$** | Cukup (*Reasonable Forecasting*) | — |
| **$> 50\%$** | Tidak Akurat (*Inaccurate*) | — |
