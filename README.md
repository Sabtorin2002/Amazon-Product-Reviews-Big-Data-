# Amazon-Product-Reviews-Big-Data
# 🧠 Big Data Project – Amazon Product Reviews

**Author:** Toma Sabin-Sebastian  
**Group:** 412  
**Dataset:** [Amazon Product Reviews – Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)

---

## 📘 Overview

This project focuses on building an **automatic review score classification system** for Amazon products using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques.  
The goal is to predict the **review score (1–5 stars)** based solely on the **textual content** of reviews.

---

## 📂 Dataset Description

The dataset consists of product reviews from Amazon, each containing user opinions and product metadata.  
Only the following columns were used for model training:

- **Summary** – short user-provided title of the review  
- **Text** – full user opinion  
- **Score** – numerical rating (1–5 stars)

Other fields such as `ProductId`, `UserId`, or `HelpfulnessNumerator` were excluded from the classification task.

---

## 🎯 Objectives

1. **Data preprocessing and cleaning** using **Pandas** and **Spark**
2. **Exploratory data analysis** and **grouping** using **SparkSQL**
3. **Training ML models** using **Naïve Bayes** and **Logistic Regression (Spark MLlib)**
4. **Training a DL model** using **LSTM (TensorFlow/Keras)**
5. **Building a reusable data pipeline**
6. **Evaluating model performance** with metrics:  
   `Accuracy`, `Precision`, `Recall`, and `F1-Score`
7. **Implementing Spark Streaming** for real-time data ingestion and processing

---

## 🧹 Data Preprocessing

Performed text cleaning with:
- Lowercasing  
- URL and punctuation removal  
- Stopword filtering  
- Lemmatization using **NLTK**

The cleaned dataset was saved as `newReviews.csv` and later split into:
- `train.csv` (80%)
- `test.csv` (20%)

---

## 🧩 Machine Learning Models (Spark MLlib)

### 🔹 Logistic Regression
- Combined `Summary` and `Text` → `combined_text`
- NLP pipeline: Tokenizer → StopWordsRemover → HashingTF → IDF
- Label encoding via `StringIndexer`
- Evaluated using standard classification metrics

### 🔹 Naïve Bayes
- Similar preprocessing pipeline as Logistic Regression
- Used **raw TF vectors** (no IDF)
- Multinomial NB with smoothing = 0.9

---

## 🧠 Deep Learning Model (TensorFlow/Keras)

### Architecture
- **Embedding layer (10,000 words)**
- **LSTM (64 units)**
- **Dropout (0.5)**
- **Dense (ReLU + Softmax output)**

### Training
- Optimizer: `Adam`
- Loss: `sparse_categorical_crossentropy`
- Epochs: `5`
- Batch size: `128`
- Validation split: `0.1`

### Observations
- Validation accuracy reached **~79%**
- Strong performance on class `5` (many samples)
- Lower recall for underrepresented classes (`2`, `4`)

---

## ⚙️ Real-Time Streaming with Spark Streaming

A simulation of streaming review data from a folder (`stream_input`) using:
- `readStream` to continuously monitor CSV files
- Batch processing via `foreachBatch`
- Real-time aggregation of:
  - total number of reviews  
  - average review score  

This component demonstrates how new data files can be processed automatically as they appear.

---

## 📊 Evaluation Metrics

| Metric     | Description                                 |
|-------------|---------------------------------------------|
| Accuracy    | Overall percentage of correct predictions   |
| Precision   | Fraction of relevant instances retrieved    |
| Recall      | Fraction of relevant instances identified   |
| F1-Score    | Harmonic mean of precision and recall       |

---

## 🧾 Technologies Used

- **Python**
- **Apache Spark (MLlib, SQL, Streaming)**
- **TensorFlow / Keras**
- **Pandas**
- **NLTK**
- **Scikit-learn**




