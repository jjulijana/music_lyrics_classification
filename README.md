
#### **Music Emotion Classification Using Lyrics**  

This project aims to classify songs based on the emotion expressed in their lyrics using machine learning models. The primary goal is to explore and compare the effectiveness of LSTM and Transformer-based approaches for emotion classification.  

---

### **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/jjulijana/music-emotion-classification.git
   cd music-emotion-classification
   ```  
2. Create a virtual environment
    ```bash
    python -m venv venv  
    source venv/bin/activate  
    ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

---

### Project structure

```
project-root/
├── data/
│   ├── raw/                # Original datasets
│   ├── preprocessed/       # Cleaned and formatted data
│   └── splits/             # Train/validation/test splits
├── utils/
│   ├── 01_data_downloading.ipynb
│   ├── 02_data_preprocessing.ipynb, preprocessing.py
│   └── 03_spliting_maxlen.ipynb, maxlen_optimizer.py, splitter.py, vocab_sizing.py
├── models/                 # Model implementation and models
│   ├── lstm/               
│   └── transformer/        
└── documentation/          
```

---

### **Key Features** 
- **Models**: LSTM and Transformer.  
- **Evaluation**: Accuracy, Precision, Recall, and F1-Score.  

---

### **Contributors**  
- [Jelena Milosevic 69/2020](https://github.com/jelena-mi)  
- [Julijana Jevtic 25/2020](https://github.com/jjulijana)  
