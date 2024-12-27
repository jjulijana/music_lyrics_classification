
#### **Music Emotion Classification Using Lyrics**  

This project aims to classify songs based on the emotion expressed in their lyrics using machine learning models. The primary goal is to explore and compare the effectiveness of LSTM and Transformer-based approaches for emotion classification.  

---

### **Project Structure**  
- **`data/`**: Contains raw and processed datasets.  
- **`models/`**: Code for building and training LSTM and Transformer models.  
- **`notebooks/`**: Jupyter notebooks for analysis and experiments.  
- **`utils/`**: Helper scripts for preprocessing, loading embeddings, and evaluation.  
- **`config/`**: Config files for model hyperparameters.  
- **`tests/`**: Unit tests for project components.  

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

### **Usage**  
1. Preprocess data:  
   ```bash
   python utils/data_preprocessing.py
   ```  
2. Train a model:  
   ```bash
   python models/lstm/lstm_training.py  # For LSTM  
   python models/transformer/transformer_training.py  # For Transformer  
   ```  
3. Evaluate performance:  
   Use the metrics provided in `utils/metrics.py`.  

---

### **Key Features**  
- **Embeddings**: Pretrained GloVe/Word2Vec for LSTM and contextual embeddings for Transformers.  
- **Models**: LSTM and Transformer (BERT-based).  
- **Evaluation**: Accuracy, Precision, Recall, and F1-Score.  

---

### **Contributors**  
Julijana Jevtic
Jelena Milosevic