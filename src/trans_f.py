import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import random
import numpy as np
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print("TensorFlow版本:", tf.__version__)
print("可用GPU设备:", tf.config.list_physical_devices('GPU'))  # 列出所有可用GPU

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 启用内存动态增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU内存动态分配已启用")
    except RuntimeError as e:
        print(e)

# 加载CSV数据集
#data_dir = Path(r'D:\大模型\transf\data')
data_dir = Path('./data')

# 读取CSV（根据实际列名修改usecols）
train_df = pd.read_csv(data_dir / "train.csv", usecols=["en", "zh"])  # 源语言列、目标语言列
val_df = pd.read_csv(data_dir / "validation.csv", usecols=["en", "zh"])

print("训练数据行数:", len(train_df))  # 预期>0，比如几百/几千行
print("验证数据行数:", len(val_df))    # 预期>0
print("训练数据前2行:\n", train_df.head(2))
# 转换为tf.data.Dataset
def df_to_dataset(df):
    # 分别提取en和zh列的numpy数组（字符串类型）
    en_texts = df["en"].astype(str).values  # 形状：(n_samples,)
    zh_texts = df["zh"].astype(str).values  # 形状：(n_samples,)
    # 从两个独立数组创建数据集，每个元素是(en_tensor, zh_tensor)
    return tf.data.Dataset.from_tensor_slices((en_texts, zh_texts))

train_examples = df_to_dataset(train_df)
val_examples = df_to_dataset(val_df)
train_samples = list(train_examples.as_numpy_iterator()) 

# 准备tokenizer
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en for en, zh in train_samples),  # 提取en列构建英文tokenizer
    target_vocab_size=2**13)
tokenizer_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(  # 变量名改为tokenizer_zh
    (zh for en, zh in train_samples),  # 提取zh列构建中文tokenizer
    target_vocab_size=2**13)

MAX_SEQ_LEN = 30
# 定义tokenizer函数
def encode(en, zh):
    # 对输入语言en编码（添加开始/结束标记）
    en_encoded = [tokenizer_en.vocab_size] + tokenizer_en.encode(en.numpy()) + [tokenizer_en.vocab_size + 1]
    # 对目标语言zh编码（添加开始/结束标记）
    zh_encoded = [tokenizer_zh.vocab_size] + tokenizer_zh.encode(zh.numpy()) + [tokenizer_zh.vocab_size + 1]
    if len(en_encoded) > MAX_SEQ_LEN:
        en_encoded = en_encoded[:MAX_SEQ_LEN]
    if len(zh_encoded) > MAX_SEQ_LEN:
        zh_encoded = zh_encoded[:MAX_SEQ_LEN]
    return en_encoded, zh_encoded  # 保持不变，确保返回两个编码结果

def tf_encode(en, zh):  # 参数名与数据集一致：(en, zh)
    result_en, result_zh = tf.py_function(encode, [en, zh], [tf.int64, tf.int64])  # 调用encode处理(en, zh)
    result_en.set_shape([None])  # 输入编码的形状
    result_zh.set_shape([None])  # 目标编码的形状
    return result_en, result_zh

# 设置缓冲区大小
BUFFER_SIZE = 20000
BATCH_SIZE = 8

# 预处理数据
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.padded_batch(BATCH_SIZE)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
    
# 示例：无位置编码的Encoder（用于消融实验）
class EncoderNoPE(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # 去掉位置编码
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

# 设置Transformer参数
num_layers = 4
d_model = 64
dff = 256
num_heads = 4
input_vocab_size = tokenizer_en.vocab_size + 2  # 输入语言是en，用tokenizer_en
target_vocab_size = tokenizer_zh.vocab_size + 2  # 目标语言是zh，用tokenizer_zh
dropout_rate = 0.1

# 创建Transformer模型
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=1000, pe_target=1000, rate=dropout_rate)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # 输出形状: (seq_len, seq_len)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

EPOCHS = 10
lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-4, decay_steps=len(train_dataset)*EPOCHS, alpha=0.01
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

# 定义train_step
@tf.function(experimental_relax_shapes=True)  # 允许输入形状动态变化
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, look_ahead_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss

checkpoint_dir = './results/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)  
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)
train_losses = []
val_losses = []
perplexities = []   

# 训练模型
for epoch in range(EPOCHS):
    total_loss = 0

    for (batch, (inp, tar)) in enumerate(train_dataset):
        batch_loss = train_step(inp, tar)
        total_loss += batch_loss

    avg_train_loss = total_loss / len(train_dataset)
    train_losses.append(avg_train_loss.numpy())  # 转numpy，避免Tensor占用内存

    val_loss = 0
    # 验证时不更新参数
    for (batch, (inp, tar)) in enumerate(val_dataset):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_pad_mask, look_ahead_mask, dec_pad_mask = create_masks(inp, tar_inp)

        preds, _ = transformer(
            inp, tar_inp, 
            training=False,
            enc_padding_mask=enc_pad_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_pad_mask
        )
        val_loss += loss_function(tar_real, preds)
    
    # 计算平均验证损失并记录
    avg_val_loss = val_loss / len(val_dataset)
    val_losses.append(avg_val_loss.numpy())

    perplexity = tf.exp(avg_val_loss).numpy()  # 困惑度=exp(平均验证损失)
    perplexities.append(perplexity)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"训练损失：{avg_train_loss:.4f} | 验证损失：{avg_val_loss:.4f} | 困惑度：{perplexity:.2f}")
    if (epoch + 1) % 5 == 0:
        save_path = checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'模型已保存到：{save_path}（第{epoch+1}个epoch）')

plt.figure(figsize=(10, 6))  # 设置图表大小，避免模糊
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', marker='o', linestyle='-', color='#1f77b4')
plt.plot(range(1, EPOCHS+1), val_losses, label='Val Loss', marker='s', linestyle='--', color='#ff7f0e')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Transformer Training & Validation Loss Curve', fontsize=14, pad=20)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)  # 添加网格，更易读
plt.savefig('./results/training_curve.png', dpi=300, bbox_inches='tight')  # 高分辨率保存
plt.close()
metrics_df = pd.DataFrame({
    'Epoch': range(1, EPOCHS+1),
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'Perplexity': perplexities
})
metrics_df.to_csv('./results/metrics.csv', index=False)

class TransformerNoPE(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, rate=0.1):
        super().__init__()
        # 使用无位置编码的 Encoder
        self.encoder = EncoderNoPE(num_layers, d_model, num_heads, dff,
                                   input_vocab_size, maximum_position_encoding=1000, rate=rate)
        # Decoder 保持原样（仍然有位置编码）
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, maximum_position_encoding=1000, rate=rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training,
                                                     look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


# 初始化“无位置编码”模型
transformer_nope = TransformerNoPE(num_layers, d_model, num_heads, dff,
                                   input_vocab_size, target_vocab_size, rate=dropout_rate)
optimizer_nope = tf.keras.optimizers.Adam(learning_rate=3e-4)

# 保存训练曲线数据
train_losses_nope, val_losses_nope, perplexities_nope = [], [], []

print("开始无位置编码模型训练 (Ablation)")

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (inp, tar)) in enumerate(train_dataset):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            preds, _ = transformer_nope(inp, tar_inp, True,
                                        enc_padding_mask, look_ahead_mask, dec_padding_mask)
            loss = loss_function(tar_real, preds)
        gradients = tape.gradient(loss, transformer_nope.trainable_variables)
        optimizer_nope.apply_gradients(zip(gradients, transformer_nope.trainable_variables))
        total_loss += loss

    avg_train_loss = total_loss / len(train_dataset)
    train_losses_nope.append(avg_train_loss.numpy())

    # 验证
    val_loss = 0
    for (batch, (inp, tar)) in enumerate(val_dataset):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_pad_mask, look_ahead_mask, dec_pad_mask = create_masks(inp, tar_inp)
        preds, _ = transformer_nope(inp, tar_inp, False,
                                    enc_pad_mask, look_ahead_mask, dec_pad_mask)
        val_loss += loss_function(tar_real, preds)
    avg_val_loss = val_loss / len(val_dataset)
    val_losses_nope.append(avg_val_loss.numpy())
    perplexity = tf.exp(avg_val_loss).numpy()
    perplexities_nope.append(perplexity)
    print(f"[NoPE] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | PPL: {perplexity:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS+1), val_losses, label='With Positional Encoding', linestyle='--', color='orange')
plt.plot(range(1, EPOCHS+1), val_losses_nope, label='Without Positional Encoding', linestyle='-', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Ablation: Effect of Positional Encoding')
plt.legend()
plt.grid(alpha=0.3)
os.makedirs('./results', exist_ok=True)
plt.savefig('./results/ablation_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n 消融实验完成：ablation_curve.png 已保存到 ./results/")


def load_model(checkpoint_dir):
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_ckpt)
    return transformer

