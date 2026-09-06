import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
import numpy as np
import itertools

def load_fashion_mnist():
    """Load and preprocess Fashion-MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return (X_train, y_train), (X_test, y_test)

def create_mlp_model(input_shape=(28, 28), hidden_units=[512, 256, 128], dropout_rate=0.2, learning_rate=0.001):
    """Factory function to build an MLP model with specified hyperparameters."""
    model = Sequential([Input(shape=input_shape), Flatten()])
    
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(10, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def create_cnn_model(input_shape=(28, 28, 1), learning_rate=0.001):
    """
    Suggested alternative approach: Convolutional Neural Network (CNN).
    CNNs preserve spatial hierarchies, edge patterns, and shape textures (like collars
    and sleeves) much better than flattened MLPs, significantly reducing confusion
    between shirts and T-shirts.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning function performing grid/random search over:
    - learning_rate
    - batch_size
    - dropout_rate
    - hidden_units architectures
    """
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [64, 128],
        'dropout_rate': [0.2, 0.3],
        'hidden_units': [
            [512, 256, 128],
            [256, 128, 64]
        ]
    }
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_val_acc = 0.0
    best_params = None
    best_model = None
    
    print(f"Starting hyperparameter tuning across {len(combinations)} configurations...")
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Testing Config {idx + 1}/{len(combinations)}: {params} ---")
        
        model = create_mlp_model(
            hidden_units=params['hidden_units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        val_acc = max(history.history['val_accuracy'])
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = model
            print(">>> New best model found!")
            
    print("\n==============================")
    print("Hyperparameter Tuning Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Parameters: {best_params}")
    print("==============================")
    
    return best_model, best_params

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    # Split training into train and validation sets
    val_split = int(0.1 * len(X_train))
    X_val, y_val = X_train[:val_split], y_train[:val_split]
    X_tr, y_tr = X_train[val_split:], y_train[val_split:]
    
    best_model, best_params = tune_hyperparameters(X_tr, y_tr, X_val, y_val)
    best_model.save("best_tuned_fashion_model.keras")
