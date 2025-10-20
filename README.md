# 📈 S&P 500 Direction Classifier  
### _Machine Learning for Financial Market Prediction_

---

## 🎯 Objectif
Prédire la **direction du S&P 500** (hausse ou baisse du lendemain) à partir d’indicateurs techniques simples.  
Ce projet montre comment appliquer des modèles de Machine Learning supervisés à des données financières **sans sur-ingénierie** : approche claire, rigoureuse et explicable.

---

## ⚙️ Méthodologie

### 1️⃣ Données
- Source : **Yahoo Finance (`yfinance`)**
- Actif : **SPY (ETF S&P 500)**  
- Période : **2010 → 2025**
- Fréquence : **Journalière**

### 2️⃣ Features
| Catégorie | Indicateurs |
|------------|-------------|
| Momentum | retards (r1 à r5) |
| Volatilité | rolling std (10j, 20j) |
| Tendances | MA10, MA20, MA ratio |
| Oscillateurs | RSI(14) |
| Target | Direction du lendemain (Up / Down) |

### 3️⃣ Modèles utilisés
- 🔹 **Logistic Regression** → baseline interprétable  
- 🔹 **Random Forest** → modèle non-linéaire plus robuste  

Les données sont découpées dans le temps (70 % train / 30 % test) pour éviter toute fuite temporelle.

---

## 📊 Évaluation

### 🧠 Metrics ML
- Accuracy  
- Precision / Recall  
- ROC-AUC  

### 💵 Metrics Stratégie
Simulation d’une stratégie simple “long/flat” :
- Sharpe Ratio  
- CAGR (rendement annualisé)  
- Max Drawdown  
- Hit Rate (pourcentage de jours positifs)  

---

## 🚀 Utilisation

### Installation
```bash
python -m venv .venv
source .venv/bin/activate      # (Windows : .venv\Scripts\activate)
pip install -r requirements.txt
