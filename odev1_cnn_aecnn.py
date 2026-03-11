"""
Ödev 1: Derin Öğrenme ile Saldırı Tespiti
NSL-KDD veri seti üzerinde CNN ve AE-CNN karşılaştırması
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ─────────────────────────────────────────────

COLUMN_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

def load_nslkdd(train_path='KDDTrain+.txt', test_path='KDDTest+.txt'):
    train = pd.read_csv(train_path, names=COLUMN_NAMES)
    test  = pd.read_csv(test_path,  names=COLUMN_NAMES)
    train.drop('difficulty', axis=1, inplace=True)
    test.drop('difficulty',  axis=1, inplace=True)
    return train, test

def preprocess(train, test):
    # Kategorik → Label Encode
    cat_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    # Etiket: normal=0, saldırı=1
    train['label'] = (train['label'] != 'normal').astype(int)
    test['label']  = (test['label']  != 'normal').astype(int)

    X_train = train.drop('label', axis=1).values.astype(np.float32)
    y_train = train['label'].values
    X_test  = test.drop('label', axis=1).values.astype(np.float32)
    y_test  = test['label'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # CNN girişi için reshape: (samples, features, 1)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_train, X_test, X_train_cnn, X_test_cnn, y_train, y_test

# ─────────────────────────────────────────────
# 2. CNN MODELİ
# ─────────────────────────────────────────────

def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ─────────────────────────────────────────────
# 3. AUTOENCODER + CNN (AE-CNN) MODELİ
# ─────────────────────────────────────────────

def build_autoencoder(input_dim, encoding_dim=32):
    inp = layers.Input(shape=(input_dim,))
    # Encoder
    encoded = layers.Dense(128, activation='relu')(inp)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inp, decoded, name='autoencoder')
    encoder     = Model(inp, encoded, name='encoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def build_aecnn(encoder, encoding_dim=32):
    """Encoder çıktısını CNN'e bağlar."""
    inp = layers.Input(shape=(encoding_dim,))
    x   = layers.Reshape((encoding_dim, 1))(inp)
    x   = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    cnn_part = Model(inp, out, name='cnn_classifier')

    # Birleşik model (encoder freeze edilebilir)
    encoder.trainable = False
    final_inp = layers.Input(shape=(encoder.input_shape[1],))
    enc_out   = encoder(final_inp)
    cnn_out   = cnn_part(enc_out)
    combined  = Model(final_inp, cnn_out, name='ae_cnn')
    combined.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return combined

# ─────────────────────────────────────────────
# 4. EĞİTİM VE DEĞERLENDİRME
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"  {model_name} – Sonuçlar")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal','Attack']))

    cm = confusion_matrix(y_test, y_pred)
    return y_pred, cm, {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred)
    }

def plot_training(history, model_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} – Eğitim Süreci', fontsize=14)

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Attack'],
                yticklabels=['Normal','Attack'])
    plt.title(f'{model_name} – Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison(metrics_cnn, metrics_aecnn, save_path=None):
    metric_names = ['accuracy','precision','recall','f1']
    cnn_vals   = [metrics_cnn[m]   for m in metric_names]
    aecnn_vals = [metrics_aecnn[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, cnn_vals,   width, label='CNN',    color='steelblue')
    bars2 = ax.bar(x + width/2, aecnn_vals, width, label='AE-CNN', color='darkorange')

    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy','Precision','Recall','F1-Score'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Skor')
    ax.set_title('CNN vs AE-CNN – Performans Karşılaştırması')
    ax.legend()

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ─────────────────────────────────────────────
# 5. ANA AKIŞ
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("NSL-KDD verisi yükleniyor...")
    train_df, test_df = load_nslkdd('KDDTrain+.txt', 'KDDTest+.txt')
    X_train, X_test, X_train_cnn, X_test_cnn, y_train, y_test = preprocess(train_df, test_df)

    n_features = X_train.shape[1]
    print(f"Özellik sayısı: {n_features} | Eğitim: {len(X_train)} | Test: {len(X_test)}")

    es = EarlyStopping(patience=5, restore_best_weights=True)

    # ── CNN ──────────────────────────────────
    print("\n[1/3] CNN modeli eğitiliyor...")
    cnn = build_cnn((n_features, 1))
    hist_cnn = cnn.fit(
        X_train_cnn, y_train,
        validation_split=0.1,
        epochs=50, batch_size=256,
        callbacks=[es], verbose=1
    )
    plot_training(hist_cnn, 'CNN', save_path='cnn_training.png')
    y_pred_cnn, cm_cnn, metrics_cnn = evaluate_model(cnn, X_test_cnn, y_test, 'CNN')
    plot_confusion_matrix(cm_cnn, 'CNN', save_path='cnn_cm.png')

    # ── Autoencoder ──────────────────────────
    print("\n[2/3] Autoencoder eğitiliyor...")
    ENCODING_DIM = 32
    ae, encoder = build_autoencoder(n_features, encoding_dim=ENCODING_DIM)
    ae.fit(
        X_train, X_train,
        validation_split=0.1,
        epochs=50, batch_size=256,
        callbacks=[es], verbose=1
    )

    # ── AE-CNN ───────────────────────────────
    print("\n[3/3] AE-CNN modeli eğitiliyor...")
    ae_cnn = build_aecnn(encoder, encoding_dim=ENCODING_DIM)
    hist_aecnn = ae_cnn.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50, batch_size=256,
        callbacks=[es], verbose=1
    )
    plot_training(hist_aecnn, 'AE-CNN', save_path='aecnn_training.png')
    y_pred_aecnn, cm_aecnn, metrics_aecnn = evaluate_model(ae_cnn, X_test, y_test, 'AE-CNN')
    plot_confusion_matrix(cm_aecnn, 'AE-CNN', save_path='aecnn_cm.png')

    # ── Karşılaştırma ─────────────────────────
    plot_comparison(metrics_cnn, metrics_aecnn, save_path='comparison.png')

    print("\nTüm grafikler kaydedildi.")
    print("Eğitim tamamlandı!")
