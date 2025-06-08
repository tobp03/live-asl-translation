import numpy as np
import json
import pandas as pd
import re

def data_loader(data_dir,
                gloss_to_index_dir, 
                filtered_txt_path=None, 
                merge_test_to_train=False
                ):
    X_train = np.load(f"{data_dir}/x_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")

    X_test = np.load(f"{data_dir}/x_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    X_val = np.load(f"{data_dir}/x_val.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)

    y_train = np.argmax(y_train, axis=1)
    y_val   = np.argmax(y_val, axis=1)
    y_test  = np.argmax(y_test, axis=1)

    decoder = None

    #If user uses partial gloss instead all the unique gloss from ASL Citizen
    if filtered_txt_path:
        #Load original gloss_to_index dictionary
        with open(gloss_to_index_dir) as f:
            gloss_to_index = json.load(f)
        
        def normalize_gloss(gloss):
            return re.sub(r'\d+', '', gloss).replace(" ", "").upper()   
        
        normalized_to_original={}

        '''
        ASL Citizen has the same gloss for different signs ex: DEAF1 and DEAF2. 
        This section groups together multiple variations of the same gloss. Ex: "GO1", "GO2" -> "GO"
        '''
        for gloss,idx in gloss_to_index.items():
            norm_gloss=normalize_gloss(gloss)
            if norm_gloss not in normalized_to_original:
                normalized_to_original[norm_gloss]=[]
            normalized_to_original[norm_gloss].append((gloss,idx))

        #Load and normalize filter glosses
        with open(filtered_txt_path) as f:
            filter_glosses = [line.strip() for line in f if line.strip()]
        top_norm = set(normalize_gloss(g) for g in filter_glosses)

        #Create subset of allowed glosses (Inner join between original glosses and filtered glosses)
        allowed_indices = set()
        for norm_gloss in top_norm:
            if norm_gloss in normalized_to_original:
                allowed_indices.update(idx for gloss, idx in normalized_to_original[norm_gloss])

        # Filter function
        def filter_data(X, y):
            mask = np.isin(y, list(allowed_indices))
            return X[mask], y[mask]
        
        # Filter datasets
        X_train, y_train= filter_data(X_train, y_train)
        X_val, y_val = filter_data(X_val, y_val)
        X_test, y_test = filter_data(X_test, y_test)

        # Generate reverse mapping from class index to gloss
        index_to_gloss = {idx: gloss for norm_gloss in top_norm
                          for gloss, idx in normalized_to_original.get(norm_gloss, []) if idx in allowed_indices}

        # Remap class indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(allowed_indices))}
        new_to_old = {v: k for k, v in old_to_new.items()}

        def remap_labels(y_sparse):
            return np.array([old_to_new[y] for y in y_sparse if y in old_to_new])
        
        y_train = remap_labels(y_train)
        y_val= remap_labels(y_val)
        y_test= remap_labels(y_test)

        # Create new decoder
        decoder = {new_idx: index_to_gloss[old_idx] for new_idx, old_idx in new_to_old.items()}
    # Merge test into train if specified
    if merge_test_to_train:
        X_train = np.concatenate([X_train, X_test], axis=0)
        y_train = np.concatenate([y_train, y_test], axis=0)
        X_test, y_test = np.array([]), np.array([])
    
    return (X_train,y_train), (X_val,y_val), (X_test,y_test), decoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def train_gru_model(X_train, y_train, X_val, y_val,X_test=None, y_test=None, save_path='best_model.keras',
                    input_shape=None, num_classes=None, batch_size=32, epochs=100):
    """
    Trains a GRU-based classification model on the given dataset.

    Parameters:
    - X_train, y_train: training data
    - X_val, y_val: validation data
    - save_path: file path to save the best model
    - input_shape: optional, override for input shape (default: inferred from X_train)
    - num_classes: optional, override for output dimension (default: inferred from y_train)
    - batch_size: training batch size
    - epochs: number of training epochs

    Returns:
    - best_model: the trained model with the best validation accuracy
    """

    # Infer shapes if not provided
    if input_shape is None:
        input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    if num_classes is None:
        num_classes = len(set(y_train))

    # Build model
    model = Sequential()
    model.add(GRU(386, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(GRU(192, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.summary()

    # Setup checkpoint to save best model
    checkpoint = ModelCheckpoint(filepath=save_path,
                                 monitor='val_sparse_categorical_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)

    # Train the model
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              callbacks=[checkpoint])

    # Load the best saved model
    best_model = load_model(save_path)

    if X_test is not None and y_test is not None:
        yhat = best_model.predict(X_test)
        ytrue = y_test
        yhat = np.argmax(yhat, axis=1).tolist()
        acc = accuracy_score(ytrue, yhat)
        f1 = f1_score(ytrue, yhat, average='macro')
        print(f"Accuracy: {acc:.4f}, F1 Score (Macro): {f1:.4f}")
        
    #Save for kaggle
    if "/kaggle/" in save_path:
        shutil.move(save_path, "/kaggle/working/best_model.keras")
    else:
        #Save for other notebooks 
        best_model.save(save_path)
    return best_model
