import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re
import os
import random
import io

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="Log Anonymizer & Generator", layout="wide")
st.title("🛡️ Log Anonymizer & Synthetic Generator")
st.markdown("""
This tool uses a **Beta-VAE** to learn log patterns and injects **Synthetic Data** (Random Names, IPs, Actions, Durations) to generate anonymized logs in JSON format.
""")


# ==============================================================================
# 2. HELPER: SYNTHETIC DATA GENERATOR
# ==============================================================================
class SyntheticDataGenerator:
    """Generates random replacement data for all fields."""

    def __init__(self):
        self.first_names = ["John", "Jane", "Alice", "Bob", "Smith", "Neo", "Trinity", "Morpheus", "James", "Mary"]
        self.last_names = ["Smith", "Doe", "Johnson", "Brown", "Anderson", "Williams", "Jones", "Miller", "Davis"]
        self.components = ["apacherexa", "nginx_secure", "auth_gateway", "proxy_sentinel", "web_shield"]

    def get_random_ip(self):
        return ".".join(map(str, (random.randint(1, 255) for _ in range(4))))

    def get_random_name(self):
        return f"{random.choice(self.first_names)}_{random.choice(self.last_names)}"

    def get_random_component(self):
        return random.choice(self.components)

    def get_random_thread_id(self):
        return f"[http-nio-8080-exec-{random.randint(1, 200)}]"

    def get_random_app_timestamp(self):
        h, m, s, ms = random.randint(0, 23), random.randint(0, 59), random.randint(0, 59), random.randint(0, 999)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def get_random_action(self):
        return random.choice(["connected", "disconnected"])

    def get_random_connection_id(self):
        return str(random.randint(1, 500))

    def get_random_duration(self):
        return str(random.randint(500, 50000))


# ==============================================================================
# 3. DATA PREPROCESSING ENGINE
# ==============================================================================
class LogPreprocessor:
    def __init__(self, max_vocab=5000, max_len=100):
        self.tokenizer = Tokenizer(num_words=max_vocab, filters='', lower=True)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.max_len = max_len

    def parse_log_line(self, line):
        # Basic parsing to extract content for training
        match = re.search(r'User "([^"]+)"', line)
        user = match.group(1) if match else "UNKNOWN_USER"

        ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line)
        ip = ip_match.group(1) if ip_match else "UNKNOWN_IP"

        # Content masking for VAE training
        clean_content = line
        if ip != "UNKNOWN_IP": clean_content = clean_content.replace(ip, "token_ip")
        if user != "UNKNOWN_USER": clean_content = clean_content.replace(user, "token_user")

        return {
            "identity": f"{ip}|{user}",
            "content": clean_content,
            "raw_log_line": line.strip()
        }

    def load_from_stream(self, file_obj):
        """Reads from Streamlit UploadedFile object"""
        data = []
        # Convert bytes to string (StringIO)
        stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))

        for line in stringio:
            parsed = self.parse_log_line(line)
            data.append(parsed)
        return pd.DataFrame(data)

    def fit_transform(self, df):
        self.tokenizer.fit_on_texts(df['content'])
        content_seq = self.tokenizer.texts_to_sequences(df['content'])
        content_vector = pad_sequences(content_seq, maxlen=self.max_len, padding='post')
        identity_labels = self.label_encoder.fit_transform(df['identity'])
        self.is_fitted = True
        return content_vector, identity_labels

    def transform(self, df):
        if not self.is_fitted: raise Exception("Fit first.")
        content_seq = self.tokenizer.texts_to_sequences(df['content'])
        content_vector = pad_sequences(content_seq, maxlen=self.max_len, padding='post')
        known_classes = set(self.label_encoder.classes_)
        df['identity_safe'] = df['identity'].apply(
            lambda x: x if x in known_classes else self.label_encoder.classes_[0])
        identity_labels = self.label_encoder.transform(df['identity_safe'])
        return content_vector, identity_labels


# ==============================================================================
# 4. MODEL ARCHITECTURE (VAE)
# ==============================================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BetaVAELossLayer(layers.Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        content_true, content_pred, z_mean_c, z_log_var_c, z_mean_s, z_log_var_s = inputs
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(content_true, content_pred), axis=-1))
        kl_c = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var_c - tf.square(z_mean_c) - tf.exp(z_log_var_c), axis=1))
        kl_s = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var_s - tf.square(z_mean_s) - tf.exp(z_log_var_s), axis=1))
        self.add_loss(recon_loss + self.beta * (kl_c + kl_s))
        return content_pred


def build_vae(vocab_size, num_identities, max_len, latent_c, latent_s, beta):
    # Encoder
    c_in = layers.Input(shape=(max_len,))
    x1 = layers.Embedding(vocab_size, 64, mask_zero=True)(c_in)
    x1 = layers.LSTM(64)(x1)
    z_mean_c = layers.Dense(latent_c)(x1)
    z_log_var_c = layers.Dense(latent_c)(x1)
    z_c = Sampling()([z_mean_c, z_log_var_c])

    s_in = layers.Input(shape=(1,))
    x2 = layers.Embedding(num_identities, 16)(s_in)
    x2 = layers.Flatten()(x2)
    z_mean_s = layers.Dense(latent_s)(x2)
    z_log_var_s = layers.Dense(latent_s)(x2)
    z_s = Sampling()([z_mean_s, z_log_var_s])

    # Decoder
    z_comb = layers.Concatenate()([z_c, z_s])
    dec_rep = layers.RepeatVector(max_len, name="decoder_repeat")
    dec_lstm = layers.LSTM(64, return_sequences=True, name="decoder_lstm")
    dec_out = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"), name="decoder_output")

    final_out = dec_out(dec_lstm(dec_rep(z_comb)))

    # Loss
    outputs = BetaVAELossLayer(beta=beta)([c_in, final_out, z_mean_c, z_log_var_c, z_mean_s, z_log_var_s])
    vae = models.Model(inputs=[c_in, s_in], outputs=outputs)
    vae.compile(optimizer='adam')

    return vae, (c_in, s_in), (z_mean_c, z_mean_s)


# ==============================================================================
# 5. STREAMLIT APP LOGIC
# ==============================================================================

# Sidebar Configuration
st.sidebar.header("Configuration")
MAX_VOCAB = st.sidebar.number_input("Max Vocab", value=5000)
MAX_LEN = st.sidebar.number_input("Max Sequence Length", value=100)
EPOCHS = st.sidebar.slider("Epochs", 1, 100, 30)
BATCH_SIZE = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
LATENT_DIM_CONTENT = 32
LATENT_DIM_STYLE = 8
BETA_VALUE = 4.0

# File Uploads
st.subheader("1. Upload Data")
col1, col2 = st.columns(2)
train_file = col1.file_uploader("Upload Training Logs (.txt)", type=["txt"])
test_file = col2.file_uploader("Upload Testing Logs (.txt)", type=["txt"])

if st.button("🚀 Train Model & Generate Logs"):
    if train_file is None or test_file is None:
        st.error("Please upload both training and testing files.")
    else:
        with st.spinner("Processing Data and Training VAE..."):

            # --- 1. Preprocessing ---
            preprocessor = LogPreprocessor(max_vocab=MAX_VOCAB, max_len=MAX_LEN)
            synthetic_gen = SyntheticDataGenerator()

            df_train = preprocessor.load_from_stream(train_file)
            X_train_c, X_train_s = preprocessor.fit_transform(df_train)

            df_test = preprocessor.load_from_stream(test_file)
            X_test_c, X_test_s = preprocessor.transform(df_test)

            vocab_size = len(preprocessor.tokenizer.word_index) + 1
            num_ids = len(preprocessor.label_encoder.classes_)

            st.success(f"Data Loaded. Vocab Size: {vocab_size}, Identities: {num_ids}")

            # --- 2. Training ---
            # We clear session to prevent clutter
            tf.keras.backend.clear_session()

            vae, inputs, z_means = build_vae(MAX_VOCAB, num_ids, MAX_LEN, LATENT_DIM_CONTENT, LATENT_DIM_STYLE,
                                             BETA_VALUE)

            # Use a progress bar for epochs (simulated or real callback)
            hist = vae.fit(
                [X_train_c, X_train_s],
                None,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )
            st.success(f"Training Complete! Final Loss: {hist.history['loss'][-1]:.4f}")

            # --- 3. Inference & Generation ---
            st.info("Generating Synthetic Logs...")

            c_in, s_in = inputs
            z_mean_c, _ = z_means
            encoder_c = models.Model(c_in, z_mean_c)

            # Get real content latents
            limit = min(200, len(df_test))
            real_z = encoder_c.predict(X_test_c[:limit], verbose=0)

            results = []

            for i in range(limit):
                # GENERATE ALL RANDOM ATTRIBUTES
                f_name = synthetic_gen.get_random_name()
                f_ip = synthetic_gen.get_random_ip()
                f_comp = synthetic_gen.get_random_component()
                f_thread = synthetic_gen.get_random_thread_id()
                f_app_ts = synthetic_gen.get_random_app_timestamp()
                f_action = synthetic_gen.get_random_action()
                f_conn_id = synthetic_gen.get_random_connection_id()

                # Logic for duration
                if f_action == "disconnected":
                    f_duration = synthetic_gen.get_random_duration()
                    msg_body = f'User "{f_name}" disconnected from connection "{f_conn_id}". Duration: {f_duration} milliseconds'
                else:
                    f_duration = "N/A"
                    msg_body = f'User "{f_name}" connected to connection "{f_conn_id}".'

                # Construct Full Log Line
                full_log = f"{f_comp}: {f_app_ts} {f_thread} INFO - {msg_body}"

                results.append({
                    'original_syslog_timestamp': 'Nov 12 12:16:46',
                    'fake_component': f_comp,
                    'fake_app_timestamp': f_app_ts,
                    'fake_thread_id': f_thread,
                    'fake_name': f_name,
                    'fake_ip': f_ip,
                    'extracted_action': f_action,
                    'extracted_target_resource': f_conn_id,
                    'extracted_duration': f_duration,
                    'generated_anonymized_msg': msg_body,
                    'generated_full_log': full_log
                })

            df_res = pd.DataFrame(results)

            # Force Column Order
            desired_order = ['original_syslog_timestamp', 'fake_component', 'fake_app_timestamp', 'fake_thread_id',
                             'fake_name', 'fake_ip', 'extracted_action', 'extracted_target_resource',
                             'extracted_duration',
                             'generated_anonymized_msg', 'generated_full_log']

            # Ensure columns exist
            for col in desired_order:
                if col not in df_res.columns: df_res[col] = "N/A"
            df_res = df_res[desired_order]

            # Rename columns
            df_res.columns = ['syslogtimestamp', 'component', 'apptime', 'threadid', 'name', 'ip', 'action', 'target',
                              'duration', 'anomalysed message', 'full log']

            # --- 4. Display & Download ---
            st.subheader("Generated Synthetic Logs")
            st.dataframe(df_res.head(10))

            # Convert to JSON for download
            json_str = df_res.to_json(orient='records', indent=4)

            st.download_button(
                label="📥 Download Logs as JSON",
                data=json_str,
                file_name="synthetic_logs.json",
                mime="application/json"
            )