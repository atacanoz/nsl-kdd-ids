"""
Ödev 2: VAE ile Anomali Tespiti
NSL-KDD veri seti – sadece normal trafik ile eğitim, anomali eşiği belirleme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score, confusion_matrix)
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
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

def load_and_preprocess(train_path='KDDTrain+.txt', test_path='KDDTest+.txt'):
    train = pd.read_csv(train_path, names=COLUMN_NAMES).drop('difficulty', axis=1)
    test  = pd.read_csv(test_path,  names=COLUMN_NAMES).drop('difficulty', axis=1)

    cat_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()
    for col in cat_cols:
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    # Binary etiket
    train['label'] = (train['label'] != 'normal').astype(int)
    test['label']  = (test['label']  != 'normal').astype(int)

    X_train_all = train.drop('label', axis=1).values.astype(np.float32)
    y_train_all = train['label'].values
    X_test      = test.drop('label', axis=1).values.astype(np.float32)
    y_test      = test['label'].values

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)
    X_test      = scaler.transform(X_test)

    # VAE sadece normal trafik ile eğitilir
    X_train_normal = X_train_all[y_train_all == 0]

    return X_train_normal, X_test, y_test, X_train_all

# ─────────────────────────────────────────────
# 2. VAE MİMARİSİ
# ─────────────────────────────────────────────

class Sampling(layers.Layer):
    """Reparameterization trick: z = μ + ε·σ"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * eps

def build_vae(input_dim, latent_dim=16):
    # ── Encoder ──────────────────────────────
    enc_inp = layers.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(128, activation='relu')(enc_inp)
    x = layers.Dense(64, activation='relu')(x)
    z_mean    = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z         = Sampling(name='z')([z_mean, z_log_var])

    encoder = Model(enc_inp, [z_mean, z_log_var, z], name='encoder')

    # ── Decoder ──────────────────────────────
    dec_inp = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(64, activation='relu')(dec_inp)
    x = layers.Dense(128, activation='relu')(x)
    dec_out = layers.Dense(input_dim, activation='linear')(x)

    decoder = Model(dec_inp, dec_out, name='decoder')

    # ── VAE ──────────────────────────────────
    class VAE(Model):
        def __init__(self, enc, dec, beta=1.0):
            super().__init__()
            self.encoder = enc
            self.decoder = dec
            self.beta    = beta

        def call(self, x):
            z_mean, z_log_var, z = self.encoder(x)
            return self.decoder(z), z_mean, z_log_var

        def train_step(self, data):
            # fit(X, X) tuple olarak gönderir, sadece girdiyi al
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                x_hat, z_mean, z_log_var = self(data, training=True)
                recon = tf.reduce_mean(tf.reduce_sum(tf.square(data - x_hat), axis=1))
                kl    = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                loss  = recon + self.beta * kl
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {'loss': loss, 'reconstruction_loss': recon, 'kl_loss': kl}

        def test_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            x_hat, z_mean, z_log_var = self(data, training=False)
            recon = tf.reduce_mean(tf.reduce_sum(tf.square(data - x_hat), axis=1))
            kl    = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            return {'loss': recon + self.beta * kl, 'reconstruction_loss': recon, 'kl_loss': kl}

    vae = VAE(encoder, decoder, beta=1.0)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder

# ─────────────────────────────────────────────
# 3. REKONSTRÜKSİYON HATASI VE EŞİK
# ─────────────────────────────────────────────

def compute_reconstruction_error(vae, X):
    x_hat, _, _ = vae(X, training=False)
    errors = np.mean((X - x_hat.numpy())**2, axis=1)
    return errors

def find_threshold(errors_normal, percentile=95):
    return np.percentile(errors_normal, percentile)

# ─────────────────────────────────────────────
# 4. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────

def plot_reconstruction_error(errors_normal, errors_test, y_test, threshold, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dağılım karşılaştırması
    axes[0].hist(errors_normal, bins=60, alpha=0.6, color='steelblue',  label='Normal (train)')
    axes[0].hist(errors_test[y_test==0], bins=60, alpha=0.6, color='green', label='Normal (test)')
    axes[0].hist(errors_test[y_test==1], bins=60, alpha=0.6, color='red',   label='Saldırı (test)')
    axes[0].axvline(threshold, color='black', linestyle='--', linewidth=2,
                    label=f'Eşik={threshold:.4f}')
    axes[0].set_xlabel('Rekonstüksiyon Hatası (MSE)')
    axes[0].set_ylabel('Frekans')
    axes[0].set_title('Rekonstrüksiyon Hatası Dağılımı')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Eşik analizi – F1 vs threshold
    thresholds = np.percentile(errors_normal, np.arange(50, 100, 1))
    f1_scores  = []
    for t in thresholds:
        pred = (errors_test >= t).astype(int)
        f1_scores.append(f1_score(y_test, pred, zero_division=0))
    axes[1].plot(thresholds, f1_scores, color='darkorange', linewidth=2)
    axes[1].axvline(threshold, color='black', linestyle='--', linewidth=2,
                    label=f'Seçilen Eşik={threshold:.4f}')
    axes[1].set_xlabel('Eşik Değeri')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('Eşik – F1 Analizi')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_roc(y_test, errors_test, save_path=None):
    fpr, tpr, _ = roc_curve(y_test, errors_test)
    auc = roc_auc_score(y_test, errors_test)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrisi – VAE Anomali Tespiti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return auc

def plot_training_loss(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],          label='Train Loss')
    axes[0].plot(history.history['val_loss'],       label='Val Loss')
    axes[0].set_title('Toplam Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['reconstruction_loss'],    label='Train Recon Loss')
    axes[1].plot(history.history['val_reconstruction_loss'], label='Val Recon Loss')
    axes[1].set_title('Rekonstrüksiyon Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle('VAE Eğitim Süreci', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ─────────────────────────────────────────────
# 5. ANA AKIŞ
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Veri hazırlanıyor...")
    X_train_normal, X_test, y_test, X_train_all = load_and_preprocess()

    print(f"Normal eğitim: {X_train_normal.shape[0]} örnek | Test: {X_test.shape[0]} örnek")
    print(f"Test saldırı oranı: {y_test.mean()*100:.1f}%")

    input_dim = X_train_normal.shape[1]
    LATENT_DIM = 16

    print("\nVAE modeli oluşturuluyor ve eğitiliyor...")
    vae, encoder, decoder = build_vae(input_dim, latent_dim=LATENT_DIM)

    history = vae.fit(
        X_train_normal, X_train_normal,   # unsupervised: input = target
        epochs=80,
        batch_size=256,
        validation_split=0.1,
        verbose=1
    )

    plot_training_loss(history, save_path='vae_training.png')

    # Rekonstürüksiyon hataları
    errors_train = compute_reconstruction_error(vae, X_train_normal)
    errors_test  = compute_reconstruction_error(vae, X_test)

    # Eşik belirleme (normal trafik %95 persentili)
    threshold = find_threshold(errors_train, percentile=95)
    print(f"\nEşik değeri (95. persentil): {threshold:.6f}")

    plot_reconstruction_error(errors_train, errors_test, y_test, threshold,
                              save_path='vae_recon_error.png')
    auc = plot_roc(y_test, errors_test, save_path='vae_roc.png')

    # Metrikler
    y_pred = (errors_test >= threshold).astype(int)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    fpr  = (y_pred[y_test==0].sum()) / (y_test==0).sum()

    print(f"\n{'='*50}")
    print("  VAE Anomali Tespiti – Sonuçlar")
    print(f"{'='*50}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"FPR       : {fpr:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Normal','Saldırı'],
                yticklabels=['Normal','Saldırı'])
    plt.title('VAE – Confusion Matrix')
    plt.ylabel('Gerçek'); plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig('vae_cm.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nTüm grafikler kaydedildi.")