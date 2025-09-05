import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import json
import os

class DataSplitter:

    def __init__(self, csv_path: str, random_state: int = 42):
        self.csv_path = csv_path
        self.random_state = random_state   # Keep 42 for reproducibility
        self.df = None
        self.label_encoder = None
        self.splits = {}
        self.class_weights = {}
        
    def load_data(self):
        print(f"Loading data from {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")
        
        required_cols = ["text", "emotion"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Dataset loaded: {len(self.df):,}")
        
        self.label_encoder = LabelEncoder()
        self.df['emotion_encoded'] = self.label_encoder.fit_transform(self.df['emotion'])
        
        return self
    
    def create_splits(
        self, 
        train_size: float = 0.8, 
        val_size: float = 0.1, 
        test_size: float = 0.1,
        min_samples_per_class: int = 10
    ):
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")
        
        if self.df is None:
            self.load_data()
        
        emotion_counts = Counter(self.df['emotion'])
        print(f"\nEmotion distribution in full dataset:")
        for emotion, count in emotion_counts.most_common():
            print(f"  {emotion}: {count:,} samples ({count/len(self.df):.3f})")
        
        min_total_needed = min_samples_per_class / min(train_size, val_size, test_size)
        problematic_classes = [emotion for emotion, count in emotion_counts.items() 
                             if count < min_total_needed]
        
        if problematic_classes:
            print(f"Warning: Classes with few samples: {problematic_classes}")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.df['text'].values,
            self.df['emotion_encoded'].values,
            test_size=test_size,
            stratify=self.df['emotion_encoded'].values,
            random_state=self.random_state
        )
        
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        self.splits = {
            'train': {'texts': X_train.tolist(), 'labels': y_train.tolist()},
            'val': {'texts': X_val.tolist(), 'labels': y_val.tolist()},
            'test': {'texts': X_test.tolist(), 'labels': y_test.tolist()}
        }
        
        print(f"\n=== Split Statistics ===")
        for split_name, split_data in self.splits.items():
            print(f"{split_name.upper()} SET: {len(split_data['texts']):,} samples")
            split_emotions = [self.label_encoder.inverse_transform([label])[0] 
                            for label in split_data['labels']]
            split_counts = Counter(split_emotions)
            
            for emotion, count in split_counts.most_common():
                proportion = count / len(split_data['texts'])
                print(f"  {emotion}: {count:,} ({proportion:.3f})")
        
        return self

    def compute_class_weights(self, method: str = 'inverse', use_split: str = 'train'):
        """
        Args:
            method: 'inverse' or 'sqrt'
            use_split: Which split to base weights on ('train', 'val', 'test', or 'full')
        """
        if use_split == 'full':
            labels = self.df['emotion_encoded'].values
        elif use_split in self.splits:
            labels = np.array(self.splits[use_split]['labels'])
        else:
            raise ValueError(f"use_split must be 'full' or one of {list(self.splits.keys())}")

        class_indices = np.arange(len(self.label_encoder.classes_))
        class_counts = np.bincount(labels, minlength=len(class_indices))
        n_samples = len(labels)
        n_classes = len(class_indices)

        if method == 'inverse':
            weights = n_samples / (n_classes * class_counts)
        elif method == 'sqrt':
            weights = np.sqrt(n_samples / (n_classes * class_counts))
        else:
            raise ValueError("method must be 'inverse' or 'sqrt'")

        if method not in self.class_weights:
            self.class_weights[method] = {}
        self.class_weights[method][use_split] = {
            'weights_dict': {self.label_encoder.classes_[i]: float(weights[i]) for i in class_indices},
            'method': method,
            'based_on': use_split
        }

        return self.class_weights[method][use_split]['weights_dict']

    def compute_all_class_weights(self, methods=['inverse', 'sqrt'], splits=['train', 'val', 'test', 'full']):
        for method in methods:
            for split in splits:
                self.compute_class_weights(method, use_split=split)

    def save_splits(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in self.splits.items():
            split_df = pd.DataFrame({
                'text': split_data['texts'],
                'emotion_encoded': split_data['labels'],
                'emotion': [self.label_encoder.inverse_transform([label])[0] 
                          for label in split_data['labels']]
            })
            
            output_path = os.path.join(output_dir, f"{split_name}.csv")
            split_df.to_csv(output_path, index=False)
            print(f"Saved {split_name} split: {output_path} ({len(split_df):,} samples)")
        
        import joblib
        encoder_path = os.path.join(output_dir, "label_encoder.joblib")
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Saved label encoder: {encoder_path}")
        
        metadata = {
            'total_samples': len(self.df),
            'emotions': list(self.label_encoder.classes_),
            'split_sizes': {split: len(data['texts']) for split, data in self.splits.items()},
            'class_weights': {
                method: {
                    split: weights_info['weights_dict']
                    for split, weights_info in splits_dict.items()
                }
                for method, splits_dict in self.class_weights.items()
            },
            'random_state': self.random_state
        }
        
        metadata_path = os.path.join(output_dir, "split_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
    def get_split_for_pytorch(self, split_name: str):
        if split_name not in self.splits:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(self.splits.keys())}")
        
        return (self.splits[split_name]['texts'], 
                self.splits[split_name]['labels'])
