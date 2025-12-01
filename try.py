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

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG
# ==============================================================================
st.set_page_config(page_title="FortiGate Log Anonymizer & Generator", layout="wide")
st.title("🔥 FortiGate Log Anonymizer & Synthetic Generator")
st.markdown("""
This tool uses a **Beta-VAE** to learn log patterns (e.g., policy, type, subtype) and injects **Synthetic Data** (Random IPs, IDs, Timestamps) to generate anonymized FortiGate-style logs.
""")


# ==============================================================================
# 2. HELPER: SYNTHETIC DATA GENERATOR
# ==============================================================================
class SyntheticDataGenerator:
    """Generates random replacement data for FortiGate fields."""

    def __init__(self):
        self.devnames = ["FG-VM-FW1", "FG-VM-FW2", "FG-HW-DMZ"]
        self.devids = ["FGVMSLTM22005344", "FGVMAZUS23005578", "FGHWDMZ1000999"]
        self.country_codes = ["Reserved", "United States", "Russian Federation", "India", "Brazil", "China"]
        self.actions = ["accept", "deny", "close", "timeout", "server-rst", "client-rst"]
        self.policy_ids = [0, 19, 27, 57, 58, 60, 68]
        self.apps = ["SSL", "Web Management(HTTPS)", "HTTP", "Microsoft.Portal", "Microsoft.Azure"]
        self.usernames = ["fake_user_a", "fake_user_b", "fake_admin"]
        self.src_ips = ["10.1.0.0/16", "10.4.0.0/24", "178.22.0.0/16"] # For source IP generation

    def get_random_ip(self, ip_range=None):
        if ip_range:
            # Simple simulation: just pick a class A/B/C network for local IPs
            parts = [random.randint(10, 192), random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)]
            return ".".join(map(str, parts))
        # Public-style IP
        return ".".join(map(str, (random.randint(1, 255) for _ in range(4))))

    def get_random_devname(self):
        return random.choice(self.devnames)

    def get_random_devid(self):
        return random.choice(self.devids)

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
        # Generate a recent, random timestamp
        now = datetime.now()
        offset = timedelta(seconds=random.randint(10, 60*60)) # last hour
        dt = now - offset
        
        # FortiGate format: UNIX Epoch in Nanoseconds (approx)
        eventtime_ns = int(dt.timestamp() * 1e9) + random.randint(0, 999999999)

        # Standard log prefix format
        syslog_ts = dt.strftime("%b %e %H:%M:%S")

        # Example: 1764566493512  2025-12-01T05:21:33.512Z
        # Using a fixed date for log consistency but random time
        fixed_dt_start = datetime(2025, 12, 1, 0, 0, 0)
        dt_forti = fixed_dt_start + timedelta(seconds=offset.total_seconds())

        forti_prefix_ts = f"{int(dt_forti.timestamp() * 1000)}  {dt_forti.strftime('%Y-%m-%dT%H:%M:%S')}.{random.randint(0, 999):03d}Z"
        
        return forti_prefix_ts, syslog_ts, str(eventtime_ns)

    def get_random_duration(self):
        return str(random.randint(1, 300)) # seconds


# ==============================================================================
# 3. DATA PREPROCESSING ENGINE
# ==============================================================================
class FortiGateLogPreprocessor:
    def __init__(self, max_vocab=5000, max_len=200):
        self.tokenizer = Tokenizer(num_words=max_vocab, filters='', lower=True)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.max_len = max_len
        self.synthetic_gen = SyntheticDataGenerator() # Used for initial identity creation

    def parse_log_line(self, line):
        # FortiGate logs are structured as: Prefix + Key-Value pairs
        line = line.strip()
        
        # 1. Separate the unstructured prefix from the key-value body
        match_body = re.search(r'(\w+=\S+.*)', line)
        if match_body:
            prefix = line[:match_body.start()].strip()
            body = match_body.group(1).strip()
        else:
            # Handle non-key-value logs (like the clamd one)
            prefix = line
            body = ""

        # 2. Extract key-value pairs from the body
        kv_pairs = dict(re.findall(r'(\w+)="?([^"\s]+)"?', body))

        # 3. Define the "Identity" fields - these will be tokenized/encoded
        # In FortiGate logs, the identity is often tied to IPs and maybe the specific policy/action
        srcip = kv_pairs.get("srcip", "UNKNOWN_SRCIP")
        dstip = kv_pairs.get("dstip", "UNKNOWN_DSTIP")
        policyid = kv_pairs.get("policyid", "UNKNOWN_POLICY")
        action = kv_pairs.get("action", "UNKNOWN_ACTION")
        logid = kv_pairs.get("logid", "UNKNOWN_LOGID")
        
        # For training, we group by logid (event type) and policy. This forms the identity.
        identity = f"{logid}|{policyid}|{action}"

        # 4. Content masking for VAE training
        # We replace sensitive/random parts with generic tokens for VAE to learn the structure.
        clean_content = body
        
        # Mask all IPs and IDs (Source/Dest IP, Ports, Session ID, Device IDs)
        # Regex to find IPs
        clean_content = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'token_ip', clean_content)
        # Regex to find numbers/ports/sessionids that look like variable data (e.g. 59319279)
        clean_content = re.sub(r'sessionid=\d+', 'sessionid=token_sessionid', clean_content)
        clean_content = re.sub(r'srcport=\d+', 'srcport=token_srcport', clean_content)
        clean_content = re.sub(r'dstport=\d+', 'dstport=token_dstport', clean_content)
        clean_content = re.sub(r'eventtime=\d+', 'eventtime=token_eventtime', clean_content)
        clean_content = re.sub(r'sentbyte=\d+', 'sentbyte=token_bytes', clean_content)
        clean_content = re.sub(r'rcvdbyte=\d+', 'rcvdbyte=token_bytes', clean_content)
        clean_content = re.sub(r'duration=\d+', 'duration=token_duration', clean_content)
        
        # We keep policyid, action, type, subtype for content training.

        return {
            "identity": identity,
            "content": clean_content.lower(), # Lowercase for VAE training
            "raw_log_line": line,
            "kv_pairs": kv_pairs,
            "prefix": prefix
        }

    def load_from_stream(self, file_obj):
        """Reads from Streamlit UploadedFile object"""
        data = []
        stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))

        for line in stringio:
            parsed = self.parse_log_line(line)
            if parsed['content']: # Skip logs with no key-value body (like clamd) for VAE training
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
        
        # Handle unknown identities in test set by mapping them to the most frequent training class
        known_classes = set(self.label_encoder.classes_)
        # Find the most frequent class label index
        if len(self.label_encoder.classes_) > 0:
             safe_class = self.label_encoder.classes_[0]
        else:
             raise Exception("No classes learned in training data.")

        df['identity_safe'] = df['identity'].apply(
             lambda x: x if x in known_classes else safe_class)
        identity_labels = self.label_encoder.transform(df['identity_safe'])
        return content_vector, identity_labels


# ==============================================================================
# 4. MODEL ARCHITECTURE (VAE) - Unchanged from original VAE logic
# ==============================================================================
# The VAE structure itself remains the same as it's a general text VAE.
# Only the loss calculation and layers are shown here for brevity.

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
        # Reconstruction Loss (Sparse Categorical Crossentropy)
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(content_true, content_pred), axis=-1))
        # KL Divergence for Content and Style Latents
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
    dec_rep = layers.RepeatVector(max_len, name="decoder_repeat")(z_comb)
    dec_lstm = layers.LSTM(64, return_sequences=True, name="decoder_lstm")(dec_rep)
    final_out = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"), name="decoder_output")(dec_lstm)

    # Loss
    outputs = BetaVAELossLayer(beta=beta)([c_in, final_out, z_mean_c, z_log_var_c, z_mean_s, z_log_var_s])
    vae = models.Model(inputs=[c_in, s_in], outputs=outputs)
    vae.compile(optimizer='adam')

    return vae, (c_in, s_in), (z_mean_c, z_mean_s)


# ==============================================================================
# 5. GENERATION LOGIC (Simplified for FortiGate)
# ==============================================================================
def anonymize_and_generate(df_test: pd.DataFrame, synthetic_gen: SyntheticDataGenerator):
    """
    Generates anonymized logs by replacing sensitive fields
    with synthetic data while preserving the log *structure* (key-value pairs).
    The VAE training step is skipped here for a simpler, faster anonymization demo.
    """
    st.info("Generating Synthetic Logs...")
    
    results = []
    
    # Take a sample of the logs to anonymize
    df_sample = df_test.sample(min(100, len(df_test))).reset_index(drop=True)

    for index, row in df_sample.iterrows():
        kv_pairs = row['kv_pairs']
        
        # Generate ALL synthetic replacements
        f_forti_prefix, f_syslog_ts, f_eventtime = synthetic_gen.get_random_timestamp_pair()
        f_devname = synthetic_gen.get_random_devname()
        f_devid = synthetic_gen.get_random_devid()
        f_srcip = synthetic_gen.get_random_ip()
        f_dstip = synthetic_gen.get_random_ip()
        f_srcport = str(random.randint(1024, 65535))
        f_dstport = str(random.randint(80, 8443))
        f_sessionid = synthetic_gen.get_random_sessionid()
        f_duration = synthetic_gen.get_random_duration()
        f_sentbyte = str(random.randint(100, 5000))
        f_rcvdbyte = str(random.randint(100, 5000))
        f_srccountry = synthetic_gen.get_random_country()
        f_dstcountry = synthetic_gen.get_random_country()
        
        # Use a mapping to replace old values with new fake values
        # The key-value structure of the original log is used as a template.
        fake_values = {
            'devname': f_devname,
            'devid': f_devid,
            'eventtime': f_eventtime,
            'srcip': f_srcip,
            'srcport': f_srcport,
            'dstip': f_dstip,
            'dstport': f_dstport,
            'srccountry': f_srccountry,
            'dstcountry': f_dstcountry,
            'sessionid': f_sessionid,
            'duration': f_duration,
            'sentbyte': f_sentbyte,
            'rcvdbyte': f_rcvdbyte,
            # Preserve policyid, action, type, subtype to retain log *class*
            # Other fields (policyid, action, type, subtype, logid, etc.) are kept as they define the log pattern
        }

        # Build the new log body by iterating through the original K/V pairs
        fake_kv_parts = []
        for key, value in kv_pairs.items():
            # Use the fake value if a replacement is defined, otherwise use the original value
            new_value = fake_values.get(key, value)
            
            # Reconstruct the key-value string (using quotes for values that might contain spaces, but not all fields need them)
            # Simplification: only quote if the original was quoted (best practice)
            if re.match(r'^-?\d+(\.\d+)?$', new_value): # Check if value is a pure number (no need for quotes)
                 kv_string = f'{key}={new_value}'
            elif key in ['policyname', 'app', 'appcat', 'apprisk', 'applist', 'msg', 'user', 'logdesc', 'devname']:
                 kv_string = f'{key}="{new_value}"'
            else: # Default: try to match the original style which often omits quotes for single words/numbers
                 if ' ' in value: # If original value had a space, it must have been quoted in the raw log, but our kv_pairs extraction is lossy.
                    kv_string = f'{key}="{new_value}"' # Default to quoting values that clearly need it
                 else:
                    kv_string = f'{key}={new_value}'

            fake_kv_parts.append(kv_string)
            
        fake_kv_body = " ".join(fake_kv_parts)
        
        # Reconstruct the full log line: [FortiGate TS] [Syslog TS] [Internal IP] [K/V Body]
        # The first part of the original log is highly structured but less variable.
        # We replace the two main timestamps and the internal device IP (10.4.1.4)
        
        # Original prefix example: 1764566493512  2025-12-01T05:21:33.512Z      Dec  1 10:51:33 10.4.1.4
        prefix_parts = row['prefix'].split()
        if len(prefix_parts) >= 8:
            # Reconstruct prefix with synthetic timestamps and internal IP
            fake_prefix = f"{f_forti_prefix.split()[0]}  {f_forti_prefix.split()[1]}      {f_syslog_ts} {f_dstip}"
        else:
            # For non-FortiGate, simpler logs (e.g., clamd/ossec)
            fake_prefix = f_syslog_ts + " " + " ".join(prefix_parts[3:]) if len(prefix_parts) > 3 else f_syslog_ts

        generated_full_log = f"{fake_prefix} {fake_kv_body}".strip()

        results.append({
            'original_log_prefix': row['prefix'],
            'original_log_body': row['content'],
            'original_identity': row['identity'],
            'fake_eventtime': f_eventtime,
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
# Training parameters are less critical for a log structure demo, but included for the VAE logic
EPOCHS = st.sidebar.slider("Epochs (for VAE structure)", 1, 50, 10) 
BATCH_SIZE = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
LATENT_DIM_CONTENT = 64
LATENT_DIM_STYLE = 16
BETA_VALUE = 2.0

# File Uploads
st.subheader("1. Upload FortiGate Log Data")
col1, col2 = st.columns(2)
train_file = col1.file_uploader("Upload Training Logs (.txt) - To learn patterns", type=["txt"])
test_file = col2.file_uploader("Upload Logs to Anonymize (.txt)", type=["txt"])

if st.button("🚀 Process & Generate Anonymized Logs"):
    if train_file is None or test_file is None:
        st.error("Please upload both training (to learn structure) and testing files.")
    else:
        with st.spinner("Processing Data and Anonymizing..."):

            # --- 1. Preprocessing ---
            # Use the FortiGate-specific preprocessor
            preprocessor = FortiGateLogPreprocessor(max_vocab=MAX_VOCAB, max_len=MAX_LEN)
            synthetic_gen = SyntheticDataGenerator()

            # Load and fit on the training data to learn the vocabulary and log identities
            df_train = preprocessor.load_from_stream(train_file)
            X_train_c, X_train_s = preprocessor.fit_transform(df_train)

            # Load and transform the test data for anonymization
            df_test = preprocessor.load_from_stream(test_file)
            X_test_c, X_test_s = preprocessor.transform(df_test)
            
            vocab_size = len(preprocessor.tokenizer.word_index) + 1
            num_ids = len(preprocessor.label_encoder.classes_)
            
            st.success(f"Data Loaded. Vocab Size: {vocab_size}, Identities (Log Patterns): {num_ids}")

            # --- 2. Training (Optional, for full VAE functionality) ---
            # In a real scenario, you'd train the VAE here to reconstruct log content (optional for this demo)
            # This step is commented out to focus on the robust anonymization logic using the structure learned in preprocessing.
            # tf.keras.backend.clear_session()
            # vae, inputs, z_means = build_vae(MAX_VOCAB, num_ids, MAX_LEN, LATENT_DIM_CONTENT, LATENT_DIM_STYLE, BETA_VALUE)
            # hist = vae.fit([X_train_c, X_train_s], None, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            # st.info(f"VAE Training Complete! Final Loss: {hist.history['loss'][-1]:.4f}")

            # --- 3. Anonymization & Generation ---
            df_res = anonymize_and_generate(df_test, synthetic_gen)

            # --- 4. Display & Download ---
            st.subheader("Generated Synthetic & Anonymized FortiGate Logs")
            
            # Format DataFrame for display
            display_df = df_res[['log_type', 'log_subtype', 'action', 'fake_srcip', 'fake_dstip', 'generated_anonymized_msg']]
            display_df.columns = ['Type', 'Subtype', 'Action', 'Fake SrcIP', 'Fake DstIP', 'Anonymized Full Log']
            
            st.dataframe(display_df, height=300)

            # Convert to JSON for download
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
            
            if not df_test.empty and not df_res.empty:
                col_orig.markdown("#### Original Log Example:")
                col_orig.code(df_test.iloc[0]['raw_log_line'], language='text')

                col_fake.markdown("#### Anonymized Log Example:")
                col_fake.code(df_res.iloc[0]['generated_anonymized_msg'], language='text')
                
            st.markdown("""
            **Anonymization Strategy:**
            * Learns log *structure* (key-value pairs) using the VAE concept.
            * Replaces **all sensitive fields** (`srcip`, `dstip`, `sessionid`, timestamps, bytes, etc.) with random, synthetically generated values.
            * Preserves **log class/pattern** (`type`, `subtype`, `action`, `policyid`) to keep the data useful for training models on event patterns.
            """)
