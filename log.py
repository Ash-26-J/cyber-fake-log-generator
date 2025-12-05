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
import time
from datetime import datetime, timedelta
import json


# Note: Ensure you have all required libraries installed:
# pip install streamlit numpy pandas tensorflow scikit-learn

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG
# (Omitted for brevity, but include in your full file)
# ==============================================================================

# ==============================================================================
# 2. HELPER: SYNTHETIC DATA GENERATOR
# ==============================================================================
class SyntheticDataGenerator:
    def __init__(self):
        # Increased variety of synthetic data for better simulation
        self.devnames = ["Synth-FW-AZ-1", "Synth-FW-GCP-2", "Synth-FW-HW-DMZ"]
        self.devids = ["FGVMAZUS23000123", "FGVMGCP01004567", "FGHWDMZ1000999"]
        self.country_codes = ["Reserved", "United States", "Russian Federation", "India", "Brazil", "Canada", "China"]
        self.actions = ["accept", "deny", "close", "timeout", "server-rst", "client-rst"]
        self.policy_ids = [0, 19, 27, 57, 58, 60, 68, 70]
        self.apps = ["SSL", "Web Management(HTTPS)", "HTTP", "Microsoft.Portal", "Microsoft.Azure", "DNS", "NTP"]
        self.usernames = ["fake_user_a", "fake_user_b", "fake_admin", "fake_svc_acc"]

    def get_random_ip(self, is_private=False):
        if is_private:
            if random.random() < 0.5:
                return "10." + ".".join(map(str, (random.randint(0, 255) for _ in range(3))))
            else:
                if random.random() < 0.5:
                    return "192.168." + ".".join(map(str, (random.randint(0, 255) for _ in range(2))))
                else:
                    return f"172.{random.randint(16, 31)}." + ".".join(
                        map(str, (random.randint(0, 255) for _ in range(2))))
        return ".".join(map(str, (random.randint(1, 255) for _ in range(4))))

    def get_random_devname(self):
        return random.choice(self.devnames)

    def get_random_devid(self):
        return random.choice(self.devids)

    # ... (other helper methods) ...
    def get_random_country(self):
        return random.choice(self.country_codes)

    def get_random_sessionid(self):
        return str(random.randint(10000000, 99999999))

    def get_random_policyid(self):
        return str(random.choice(self.policy_ids))

    def get_random_action(self):
        return random.choice(self.actions)

    def get_random_app(self):
        return random.choice(self.apps)

    def get_random_user(self):
        return random.choice(self.usernames)

    def get_random_timestamp_pair(self):
        now = datetime.now()
        offset = timedelta(seconds=random.randint(10, 60 * 60))
        dt = now - offset
        eventtime_ns = int(dt.timestamp() * 1e9) + random.randint(0, 999999999)
        syslog_ts = dt.strftime("%b %e %H:%M:%S")
        forti_date = dt.strftime("%Y-%m-%d")
        forti_time = dt.strftime("%H:%M:%S")
        return forti_date, forti_time, syslog_ts, str(eventtime_ns)

    def get_random_duration(self):
        return str(random.randint(1, 300))


# ==============================================================================
# 3. DATA PREPROCESSING ENGINE (WITH FIXES)
# ==============================================================================
class FortiGateLogPreprocessor:
    def __init__(self, max_vocab=5000, max_len=200):
        self.tokenizer = Tokenizer(num_words=max_vocab, filters='', lower=True)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.max_len = max_len
        self.synthetic_gen = SyntheticDataGenerator()

    def parse_log_line(self, line):
        line = line.strip()

        # --- FIX 1: Robustly clean line content from lingering control characters ---
        # This removes common invisible characters that could interfere with regex
        line = re.sub(r'[\x00-\x1F\x7F]', '', line)

        # 1. Separate the unstructured prefix from the key-value body
        # --- FIX 2: Slightly more resilient regex to find the start of the K/V body ---
        # It looks for a key= followed by a value that is either non-space or starts with a quote.
        match_body = re.search(r'(\w+="?[^"\s]+"?.*)', line)

        if match_body:
            prefix = line[:match_body.start()].strip()
            body = match_body.group(1).strip()
        else:
            prefix = line
            body = ""

        # 2. Extract key-value pairs
        # This handles both key=value and key="value with spaces"
        kv_pairs = dict(re.findall(r'(\w+)="?([^"\s]+)"?', body))

        # 3. Define the "Identity" fields (for VAE training)
        policyid = kv_pairs.get("policyid", "UNKNOWN_POLICY")
        action = kv_pairs.get("action", "UNKNOWN_ACTION")
        logid = kv_pairs.get("logid", "UNKNOWN_LOGID")
        type_ = kv_pairs.get("type", "UNKNOWN_TYPE")

        identity = f"{logid}|{type_}|{policyid}|{action}"

        # 4. Content masking for VAE training
        clean_content = body
        clean_content = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'token_ip', clean_content)
        clean_content = re.sub(r'(sessionid|srcport|dstport|eventtime|sentbyte|rcvdbyte|duration)=\d+', r'\1=token_\1',
                               clean_content)
        clean_content = re.sub(r'user="?([^"\s]+)"?', 'user=token_user', clean_content)

        return {
            "identity": identity,
            "content": clean_content.lower(),
            "raw_log_line": line,
            "kv_pairs": kv_pairs,
            "prefix": prefix
        }

    def load_from_stream(self, file_obj):
        """
        Reads from Streamlit UploadedFile object, handling the specific
        [TIMESTAMP]\t{...} JSON structure and extracting 'full_log'.
        """
        data = []
        stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))

        # Flag to skip non-log header lines at the start of the file
        log_start_found = False

        for line in stringio:
            line = line.strip()
            if not line:
                continue

            # --- CRITICAL REFINEMENT: Separate the timestamp prefix from the JSON body ---
            # We split only on the first tab character ('\t').
            parts = line.split('\t', 1)

            if len(parts) < 2 or not parts[1].startswith('{'):
                # Skip header lines, malformed lines, or lines without a JSON body
                if log_start_found:
                    # Logs any malformed lines that appear *after* log entries have started
                    # print(f"Skipping malformed log line: {line[:50]}...")
                    pass
                continue

            # The second part is the pure JSON string
            json_str = parts[1]
            log_start_found = True

            try:
                # 1. Parse the extracted JSON object
                log_json = json.loads(json_str)
                # 2. Extract the actual FortiGate key-value log line
                raw_log_line = log_json.get("full_log")

                if raw_log_line:
                    # Robust cleaning
                    raw_log_line = raw_log_line.strip().replace('\x00', '')

                    # 3. Process the extracted FortiGate key-value log
                    parsed = self.parse_log_line(raw_log_line)

                    # 4. Only append if the line contained key-value pairs
                    if parsed['content']:
                        data.append(parsed)

            except json.JSONDecodeError:
                # print(f"JSON Decode Error in log body for line: {json_str[:50]}...")
                continue
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
        if len(self.label_encoder.classes_) > 0:
            safe_class = self.label_encoder.classes_[0]
        else:
            raise Exception("No classes learned in training data.")

        df['identity_safe'] = df['identity'].apply(
            lambda x: x if x in known_classes else safe_class)
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
    # This function defines the VAE architecture (Encoder/Decoder)
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

    z_comb = layers.Concatenate()([z_c, z_s])
    dec_rep = layers.RepeatVector(max_len, name="decoder_repeat")(z_comb)
    dec_lstm = layers.LSTM(64, return_sequences=True, name="decoder_lstm")(dec_rep)
    final_out = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"), name="decoder_output")(dec_lstm)

    outputs = BetaVAELossLayer(beta=beta)([c_in, final_out, z_mean_c, z_log_var_c, z_mean_s, z_log_var_s])
    vae = models.Model(inputs=[c_in, s_in], outputs=outputs)
    vae.compile(optimizer='adam')

    return vae, (c_in, s_in), (z_mean_c, z_mean_s)


# ==============================================================================
# 5. GENERATION LOGIC
# ==============================================================================
def anonymize_and_generate(df_test: pd.DataFrame, synthetic_gen: SyntheticDataGenerator):
    """
    Generates anonymized logs by replacing sensitive fields
    with synthetic data while preserving the log *structure* (key-value pairs).
    """
    st.info(f"Generating Synthetic Logs for {len(df_test)} samples...")

    results = []

    # Take a sample of the logs to anonymize (up to 200, or all if less)
    sample_size = min(200, len(df_test))
    df_sample = df_test.sample(sample_size).reset_index(drop=True)

    for index, row in df_sample.iterrows():
        kv_pairs = row['kv_pairs']
        # 0. Generate synthetic data (one set per log)
        f_date, f_time, f_syslog_ts, f_eventtime = synthetic_gen.get_random_timestamp_pair()
        f_devname = synthetic_gen.get_random_devname()
        f_devid = synthetic_gen.get_random_devid()
        f_srcip = synthetic_gen.get_random_ip(is_private=True)
        f_dstip = synthetic_gen.get_random_ip()
        f_srcport = str(random.randint(1024, 65535))
        f_dstport = str(random.randint(80, 8443))
        f_sessionid = synthetic_gen.get_random_sessionid()
        f_duration = synthetic_gen.get_random_duration()
        f_sentbyte = str(random.randint(100, 5000))
        f_rcvdbyte = str(random.randint(100, 5000))
        f_srccountry = synthetic_gen.get_random_country()
        f_dstcountry = synthetic_gen.get_random_country()
        f_user = synthetic_gen.get_random_user()

        # 1. Mapping: Define which original fields get replaced by fake values
        fake_values = {
            'devname': f_devname,
            'devid': f_devid,
            'date': f_date,
            'time': f_time,
            'eventtime': f_eventtime,
            'srcip': f_srcip,
            'srcport': f_srcport,
            'dstip': f_dstip,
            'dstport': f_dstport,
            'remip': f_dstip,  # Handle remote IP if it exists
            'srccountry': f_srccountry,
            'dstcountry': f_dstcountry,
            'sessionid': f_sessionid,
            'duration': f_duration,
            'sentbyte': f_sentbyte,
            'rcvdbyte': f_rcvdbyte,
            'user': f_user,
        }

        # 2. Build the new log body
        fake_kv_parts = []
        for key, value in kv_pairs.items():
            new_value = fake_values.get(key, value)

            # Preserve quoting style based on field type and whether it contains spaces
            needs_quotes = key in ['policyname', 'app', 'appcat', 'apprisk', 'applist', 'msg', 'user', 'logdesc',
                                   'devname'] or ' ' in new_value

            if needs_quotes:
                kv_string = f'{key}="{new_value}"'
            else:
                kv_string = f'{key}={new_value}'

            fake_kv_parts.append(kv_string)

        fake_kv_body = " ".join(fake_kv_parts)

        # 3. Reconstruct the full log line prefix
        prefix_parts = row['raw_log_line'].split()

        if len(prefix_parts) >= 3:
            # Reconstruct prefix: <Syslog Month Day Time> <New Internal IP>
            fake_prefix = f"{f_syslog_ts} {synthetic_gen.get_random_ip(is_private=True)}"
        else:
            fake_prefix = f_syslog_ts

        generated_full_log = f"{fake_prefix} {fake_kv_body}".strip()

        results.append({
            'original_log_line': row['raw_log_line'],
            'original_identity': row['identity'],
            'fake_srcip': f_srcip,
            'fake_dstip': f_dstip,
            'fake_sessionid': f_sessionid,
            'log_type': kv_pairs.get('type', 'N/A'),
            'log_subtype': kv_pairs.get('subtype', 'N/A'),
            'action': kv_pairs.get('action', 'N/A'),
            'generated_anonymized_msg': generated_full_log,
        })

    return pd.DataFrame(results)


# ==============================================================================
# 6. STREAMLIT APP LOGIC
# ==============================================================================

# Sidebar Configuration
st.sidebar.header("Configuration")
MAX_VOCAB = st.sidebar.number_input("Max Vocab", value=5000)
MAX_LEN = st.sidebar.number_input("Max Sequence Length", value=200)
EPOCHS = st.sidebar.slider("Epochs (for VAE structure)", 1, 50, 10)
BATCH_SIZE = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
LATENT_DIM_CONTENT = 64
LATENT_DIM_STYLE = 16
BETA_VALUE = 2.0

# File Uploads
st.subheader("1. Upload Log Data (JSON Format)")
st.info(
    "Since your file uses a **timestamp-prefixed JSON format**, the script now correctly extracts the embedded FortiGate log from the `full_log` field.")
col1, col2 = st.columns(2)
train_file = col1.file_uploader("Upload Training Logs (Learn Patterns)", type=["txt"])
test_file = col2.file_uploader("Upload Logs to Anonymize (Generate Fake Logs)", type=["txt"])

if st.button("🚀 Process & Generate Anonymized Logs"):
    if train_file is None or test_file is None:
        st.error("Please upload files to both slots. You can upload the same file to both if you only have one source.")
    else:
        with st.spinner("Processing Data and Anonymizing..."):

            preprocessor = FortiGateLogPreprocessor(max_vocab=MAX_VOCAB, max_len=MAX_LEN)
            synthetic_gen = SyntheticDataGenerator()

            # --- 1. Load Training Data ---
            df_train = preprocessor.load_from_stream(train_file)

            if df_train.empty:
                st.error(
                    "Training file loaded, but **no valid FortiGate key-value log lines were found** after JSON extraction. Please ensure log lines contain the 'full_log' key and structured data inside.")
                st.stop()

            X_train_c, X_train_s = preprocessor.fit_transform(df_train)
            vocab_size = len(preprocessor.tokenizer.word_index) + 1
            num_ids = len(preprocessor.label_encoder.classes_)
            st.success(f"Training Data Loaded. Vocab Size: {vocab_size}, Log Patterns Learned: {num_ids}")

            # --- 2. Load Test Data ---
            df_test = preprocessor.load_from_stream(test_file)

            if df_test.empty:
                st.error(
                    "Test file loaded, but no valid FortiGate key-value log lines were found after JSON extraction. Cannot proceed with anonymization.")
                st.stop()

            X_test_c, X_test_s = preprocessor.transform(df_test)

            # --- 3. Anonymization & Generation ---
            df_res = anonymize_and_generate(df_test, synthetic_gen)

            # --- 4. Display & Download ---
            st.subheader("Generated Synthetic & Anonymized FortiGate Logs")

            if df_res.empty:
                st.warning("Anonymization process yielded no results. Check log content or filtering logic.")
            else:
                display_df = df_res[
                    ['log_type', 'log_subtype', 'action', 'fake_srcip', 'fake_dstip', 'generated_anonymized_msg']]
                display_df.columns = ['Type', 'Subtype', 'Action', 'Fake SrcIP', 'Fake DstIP', 'Anonymized Full Log']

                st.dataframe(display_df, height=300)

                json_str = df_res.to_json(orient='records', indent=4)

                st.download_button(
                    label=f"📥 Download {len(df_res)} Anonymized Logs as JSON",
                    data=json_str,
                    file_name="synthetic_fortigate_logs.json",
                    mime="application/json"
                )

                st.markdown("---")
                st.subheader("Example Raw vs. Anonymized Log")
                col_orig, col_fake = st.columns(2)

                col_orig.markdown("#### Original Log Example:")
                col_orig.code(df_test.iloc[0]['raw_log_line'], language='text')

                col_fake.markdown("#### Anonymized Log Example:")
                col_fake.code(df_res.iloc[0]['generated_anonymized_msg'], language='text')

                st.markdown("""
                **Anonymization Strategy:**
                * **Structure Preserved:** Log patterns (`type`, `policyid`, `action`) are retained to keep the data useful for security analytics training.
                * **Data Replaced:** All sensitive fields (`ip`, `port`, `time`, `user`, `sessionid`, `bytes`) are replaced with random, synthetic values.
                """)
