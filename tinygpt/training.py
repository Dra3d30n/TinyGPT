import os


import numpy as host_np

import euclid as Euclid
from euclid.backend import xp as np

from transformers import AutoTokenizer
import threading
import time
from tinygpt import TinyGPT

TRAIN_STEPS = 200000

_last_step = 0
_training_done = False

def heartbeat():
    while not _training_done:
        print(
            f"[HEARTBEAT] Training alive | "
            f"step={_last_step}/{TRAIN_STEPS} | "
            f"time={time.strftime('%H:%M:%S')}",
            flush=True,
        )
        time.sleep(100)

heartbeat_thread = threading.Thread(
    target=heartbeat,
    daemon=True,
)

heartbeat_thread.start()
# ============================================================
# CONFIGURATION
# ============================================================
starting_iteration=100000
TEACHER_MODEL = "mistralai/Mistral-7B-v0.1"
TINYGPT_PATH = "current_model/tinygpt_weights.npz"

# ------------------------------------------------------------
# Student
# ------------------------------------------------------------

D_MODEL = 768
NUM_HEADS = 12
NUM_BLOCKS = 12
FF_DIM = 3072

SEQ_LEN = 256

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

# Stream FineWeb-Edu instead of using the old precomputed
# philosophy .npz distillation dataset.
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"

# Number of teacher tokens retained per position
TOP_K = 128

# Number of training batches.
SHUFFLE_BUFFER = 100000

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 6
#EPOCHS = 50000

LEARNING_RATE = 1e-4

TEMPERATURE = 2.0

KD_WEIGHT = 1.0
CE_WEIGHT = 0.1

SEED = 10


# ============================================================
# RANDOM SEEDS
# ============================================================

Euclid.xp.random.seed(SEED)
np.random.seed(SEED)



# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL
)


# ============================================================
# TEACHER
# ============================================================

print("\nLoading teacher...")

VOCAB_SIZE = tokenizer.vocab_size

print(
    "Tokenizer vocabulary:",
    tokenizer.vocab_size
)

print(
    "Teacher vocabulary:",
    VOCAB_SIZE
)


# ============================================================
# LOAD FINEWEB-EDU
# ============================================================

print("\nLoading FineWeb-Edu...")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "Install the Hugging Face datasets package first: "
        "pip install datasets"
    )

# Streaming avoids downloading the entire corpus.
fineweb = load_dataset(
    DATASET_NAME,
    name=DATASET_CONFIG,
    split="train",
    streaming=True,
)

fineweb = fineweb.shuffle(
    seed=SEED,
    buffer_size=SHUFFLE_BUFFER,
)

print("FineWeb-Edu streaming dataset ready.")


# ============================================================
# TOKEN STREAM
# ============================================================

def token_sequence_stream():
    """
    Turn FineWeb-Edu documents into fixed-length LM examples.

    inputs  = tokens[0:SEQ_LEN]
    targets = tokens[1:SEQ_LEN+1]
    """
    buffer = []

    for example in fineweb:
        text = example.get("text", "")

        if not text or len(text.strip()) < 20:
            continue

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(token_ids) < 2:
            continue

        buffer.extend(token_ids)

        while len(buffer) >= SEQ_LEN + 1:
            chunk = buffer[:SEQ_LEN + 1]
            del buffer[:SEQ_LEN + 1]

            yield (
                np.asarray(chunk[:-1], dtype=np.int64),
                np.asarray(chunk[1:], dtype=np.int64),
            )


def get_batch(stream, batch_size):
    xs = []
    ys = []

    while len(xs) < batch_size:
        x, y = next(stream)
        xs.append(x)
        ys.append(y)

    return (
        np.asarray(xs, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
    )


# ============================================================
# VOCABULARY
# ============================================================

VOCAB_SIZE = int(tokenizer.vocab_size)

print("Student vocabulary:", VOCAB_SIZE)




# ============================================================
# CREATE STUDENT
# ============================================================

print("\nCreating student...")

student=TinyGPT.load(TINYGPT_PATH)
# student = TinyGPT(
#     VOCAB_SIZE,
#     SEQ_LEN,
#     D_MODEL,
#     NUM_HEADS,
#     NUM_BLOCKS,
#     FF_DIM,
# )
print(
    "Student parameter tensors:",
    len(student.parameters())
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Euclid.optimizers.Adam(
    learning_rate=LEARNING_RATE,
)

optimizer.set_parameters(
    student.parameters()
)


cross_entropy = (
    Euclid.losses.CrossEntropy()
)
def sparse_cross_entropy(logits, targets):
    """
    logits: Tensor [N, V]
    targets: integer array [N]
    """

    targets = np.asarray(targets, dtype=np.int64)

    max_logits = logits.data.max(axis=-1, keepdims=True)
    shifted = logits.data - max_logits

    exp_logits = np.exp(shifted)
    sum_exp = exp_logits.sum(axis=-1, keepdims=True)

    log_probs = shifted - np.log(sum_exp)

    loss_data = -log_probs[
        np.arange(targets.shape[0]),
        targets
    ].mean()

    loss = Euclid.Tensor(
        np.asarray(loss_data, dtype=np.float32),
        requires_grad=True
    )

    def backward():
        grad = exp_logits / sum_exp

        grad[
            np.arange(targets.shape[0]),
            targets
        ] -= 1

        grad /= targets.shape[0]

        logits.grad = grad

    loss._backward = backward
    loss._prev = {logits}

    return loss


# ============================================================
# NUMPY SOFTMAX
# ============================================================

def softmax_np(x):

    x = (
        x
        - np.max(
            x,
            axis=-1,
            keepdims=True,
        )
    )

    exp_x = np.exp(x)

    return (
        exp_x
        /
        np.sum(
            exp_x,
            axis=-1,
            keepdims=True,
        )
    )


# ============================================================
# TRAIN
# ============================================================


print("\nStarting FineWeb-Edu training...\n", flush=True)

stream = token_sequence_stream()

for epoch in range(TRAIN_STEPS):

    _last_step = epoch

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    batch_x, batch_y = get_batch(
        stream,
        BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    x = Euclid.Tensor(batch_x)
    student_logits = student(x)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    student_flat = student_logits.reshape(
        -1,
        VOCAB_SIZE,
    )

    target_ids = batch_y.reshape(-1)

    loss = sparse_cross_entropy(
    student_flat,
    target_ids,
    )   

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    optimizer.zero_grad()
    #print(f"VRAM: {np.get_default_memory_pool().used_bytes() / 1024**3:.2f} GB")
    loss.backward()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{TRAIN_STEPS} | "
        f"Loss: {float(loss.data):.6f}",
        flush=True,
    )
    

    # --------------------------------------------------------
    # Checkpoint every 1000 epochs
    # --------------------------------------------------------

    if (epoch + 1) % 1000 == 0:
        student.save(
            f"tiny_gpt_distilled_step_{epoch + 1 +starting_iteration}.npz"
        )

        print(
            f"Saved checkpoint at epoch {epoch + 1+starting_iteration}",
            flush=True,
        )

_training_done = True


print("\nTraining complete.", flush=True)