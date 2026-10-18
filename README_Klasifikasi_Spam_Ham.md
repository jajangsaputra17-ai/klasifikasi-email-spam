# Klasifikasi Pesan Spam dan Ham Menggunakan Machine Learning

## Deskripsi Proyek
Proyek ini dibuat untuk mengklasifikasikan pesan teks menjadi dua kategori, yaitu **spam** dan **ham** (pesan normal). Klasifikasi dilakukan dengan teknik pemrosesan teks dan algoritma Machine Learning. Model yang digunakan dan dibandingkan adalah **Multinomial Naive Bayes** dan **Logistic Regression**.

Proyek ini dapat membantu memahami alur dasar klasifikasi teks: memuat dataset, memeriksa data, membersihkan teks, mengubah teks menjadi fitur numerik, melatih model, mengevaluasi hasil, dan mencoba prediksi pada pesan baru.

## Dataset
Dataset yang digunakan adalah `dataset_spam_ham_1600.csv`.

Berdasarkan pemeriksaan file dataset yang diberikan:
- **Jumlah baris:** 1.600 pesan
- **Jumlah kolom:** 3
- **Kolom:** `id`, `text`, dan `label`
- **Label spam:** 800 data
- **Label ham:** 800 data
- **Nilai kosong:** tidak ditemukan pada ketiga kolom
- **Baris duplikat:** tidak ditemukan berdasarkan pemeriksaan seluruh kolom

### Penjelasan kolom
| Kolom | Penjelasan |
|---|---|
| `id` | Nomor identitas setiap baris data. |
| `text` | Isi pesan yang digunakan sebagai bahan klasifikasi. |
| `label` | Kategori pesan, yaitu `spam` atau `ham`. |

**Spam** adalah pesan yang tidak diinginkan, misalnya pesan promosi berlebihan, hadiah/undian, atau ajakan mengeklik tautan. **Ham** adalah pesan normal yang bukan spam, misalnya pemberitahuan rapat, invoice, atau pengingat dokumen. Contoh tersebut menjelaskan kategori secara umum; label yang dipakai model mengikuti isi kolom `label` di dataset.

Jumlah spam dan ham pada dataset ini seimbang, masing-masing 800 pesan. Hal ini berarti kedua kelas memiliki jumlah data yang sama sebelum pembagian data latih dan data uji.

## Teknologi dan Library
- **Python** untuk bahasa pemrograman.
- **Pandas** untuk membaca dan mengolah dataset.
- **NumPy** untuk kebutuhan komputasi numerik.
- **re** untuk membersihkan teks menggunakan pola regular expression.
- **Matplotlib** dan **Seaborn** untuk visualisasi jumlah label dan confusion matrix.
- **Scikit-learn** untuk membagi dataset, mengubah teks menjadi fitur TF-IDF, melatih model, dan mengevaluasi prediksi.

## Alur Program

1. **Memuat dataset**  
   Dataset CSV dibaca menggunakan `pandas.read_csv()`. Program menampilkan informasi kolom, jumlah setiap label, dan beberapa baris awal.

2. **Memeriksa distribusi label**  
   Program menghitung jumlah spam dan ham serta menampilkannya dalam grafik batang.

3. **Membersihkan teks**  
   Fungsi `bersihkan_teks()` mengubah huruf menjadi huruf kecil, menghapus angka dan tanda baca, serta merapikan spasi. Hasilnya disimpan dalam kolom `clean_text`.

4. **Mengubah label menjadi angka**  
   Label `spam` diubah menjadi `1`, sedangkan `ham` menjadi `0`. Nilai numerik ini disimpan pada kolom `target`.

5. **Membagi dataset**  
   Data dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split()`. Parameter `stratify=y` menjaga proporsi kelas agar tetap terwakili pada kedua bagian, sedangkan `random_state=42` membuat pembagian dapat diulang.

6. **Mengubah teks menjadi fitur TF-IDF**  
   `TfidfVectorizer(max_features=3000)` mengubah teks menjadi representasi numerik berdasarkan bobot kata. Vectorizer dipelajari hanya dari data latih, kemudian diterapkan ke data uji.

7. **Melatih model Naive Bayes**  
   `MultinomialNB` dilatih menggunakan fitur TF-IDF dan label data latih. Model kemudian digunakan untuk memprediksi data uji.

8. **Melatih model Logistic Regression**  
   Logistic Regression dilatih dan diuji menggunakan pembagian dataset dan fitur yang sama agar hasilnya dapat dibandingkan.

9. **Mengevaluasi model**  
   Program menampilkan akurasi, precision, recall, F1-score, dan confusion matrix. Nilai evaluasi harus diambil dari output saat notebook dijalankan, bukan diperkirakan sebelumnya.

10. **Menguji pesan baru**  
    Fungsi `cek_email()` membersihkan pesan baru, mengubahnya dengan vectorizer yang sudah dilatih, lalu memprediksi apakah pesan tersebut termasuk spam atau ham. Pada notebook ini, fungsi prediksi menggunakan model Naive Bayes.

## Cara Menjalankan
1. Unduh atau salin file notebook `Klasifikasi_Spam_Ham_dengan_Deskripsi.ipynb`.
2. Simpan notebook dan file dataset dalam folder yang sama.
3. Ubah nama file dataset menjadi **`dataset_spam_ham_1600.csv`** jika nama file hasil unduhan masih `dataset_spam_ham_1600(1).csv`, karena kode notebook membaca nama file tanpa `(1)`.
4. Buka folder tersebut melalui Jupyter Notebook atau JupyterLab.
5. Jalankan setiap sel dari atas ke bawah secara berurutan.
6. Periksa output jumlah data, grafik, metrik evaluasi, confusion matrix, dan hasil pengujian pesan baru.

Jika library belum tersedia, jalankan perintah berikut di terminal atau sel notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Pembagian Data
Dengan 1.600 baris dan pembagian 80:20, secara umum terdapat **1.280 data latih** dan **320 data uji**. Karena dataset memiliki 800 spam dan 800 ham serta menggunakan `stratify=y`, pembagian yang dihasilkan diharapkan mempertahankan proporsi kedua kelas.

## Evaluasi
Metrik yang digunakan:
- **Accuracy:** proporsi seluruh prediksi yang benar.
- **Precision:** proporsi prediksi suatu kelas yang benar.
- **Recall:** proporsi data sebenarnya pada suatu kelas yang berhasil ditemukan model.
- **F1-score:** ringkasan keseimbangan precision dan recall.
- **Confusion matrix:** memperlihatkan jumlah prediksi benar dan salah untuk masing-masing kelas.

Untuk membahas hasil, bandingkan nilai kedua model dan perhatikan kesalahan ketika pesan spam diprediksi sebagai ham serta pesan ham diprediksi sebagai spam. Masukkan angka aktual dari output notebook ke laporan setelah program dijalankan.

## Batasan
Pembersihan teks dalam proyek ini masih sederhana karena menghapus angka dan tanda baca. Pada beberapa pesan, angka atau simbol dapat membawa informasi penting. Hasil model juga bergantung pada contoh yang tersedia di dataset, sehingga prediksi tidak selalu benar untuk semua pesan baru.

## Kesimpulan
Proyek ini menerapkan tahapan dasar klasifikasi teks untuk membedakan pesan spam dan ham menggunakan TF-IDF, Multinomial Naive Bayes, dan Logistic Regression. Kinerja model dievaluasi menggunakan beberapa metrik klasifikasi dan confusion matrix. Kesimpulan akhir mengenai model yang lebih baik ditentukan berdasarkan hasil evaluasi aktual setelah seluruh sel notebook dijalankan.

## Struktur Repository yang Disarankan
```text
klasifikasi-spam-ham/
├── Klasifikasi_Spam_Ham_dengan_Deskripsi.ipynb
├── dataset_spam_ham_1600.csv
└── README.md
```
