import os
import threading
import time

import euclid as Euclid
from euclid.backend import xp as np

from transformers import AutoTokenizer
from datasets import load_dataset

from tinygpt import TinyGPT


# ============================================================
# TRAINING CONFIG
# ============================================================

TRAIN_STEPS = 50000

STARTING_ITERATION = 0

CHECKPOINT = (
    "models/TinyGPT1.5Pretrain.npz"
)

TOKENIZER_MODEL = (
    "mistralai/Mistral-7B-v0.1"
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATASET_NAME = "Open-Orca/OpenOrca"

SHUFFLE_BUFFER = 10000

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

SEQ_LEN = 256

BATCH_SIZE = 4

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

LEARNING_RATE = 5e-5

SEED = 10

# ------------------------------------------------------------
# Context allocation
# ------------------------------------------------------------
#
# We deliberately reserve space for BOTH the question
# and answer.
#
# Maximum format:
#
# <|user|>                 1 token
# question               128 tokens
# <|assistant|>             1 token
# answer                  124 tokens
# <|end|>                  1 token
#
# Total = 255 tokens
#
# One extra position is available because inputs/targets
# are shifted.
#

MAX_QUESTION_TOKENS = 128
MAX_ANSWER_TOKENS = 124


# ============================================================
# HEARTBEAT
# ============================================================

_last_step = 0
_training_done = False


def heartbeat():

    while not _training_done:

        print(
            f"[HEARTBEAT] Fine-tuning alive | "
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
# RANDOM SEEDS
# ============================================================

Euclid.xp.random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL
)

# Use len(tokenizer), NOT tokenizer.vocab_size.
#
# len(tokenizer) represents the actual number of IDs that
# can be produced by the tokenizer, including any existing
# added tokens.

OLD_VOCAB_SIZE = len(tokenizer)

print(
    "Original tokenizer size:",
    OLD_VOCAB_SIZE,
)


# ============================================================
# ADD SPECIAL TOKENS
# ============================================================

print("\nAdding special tokens...")

special_tokens = {
    "additional_special_tokens": [
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    ]
}

num_added = tokenizer.add_special_tokens(
    special_tokens
)

print(
    "Special tokens added:",
    num_added,
)


USER_TOKEN = tokenizer.convert_tokens_to_ids(
    "<|user|>"
)

ASSISTANT_TOKEN = tokenizer.convert_tokens_to_ids(
    "<|assistant|>"
)

END_TOKEN = tokenizer.convert_tokens_to_ids(
    "<|end|>"
)

NEW_VOCAB_SIZE = len(tokenizer)
VOCAB_SIZE = NEW_VOCAB_SIZE

print(
    "USER token:",
    USER_TOKEN,
)

print(
    "ASSISTANT token:",
    ASSISTANT_TOKEN,
)

print(
    "END token:",
    END_TOKEN,
)

print(
    "New tokenizer size:",
    NEW_VOCAB_SIZE,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading pretrained model...")

student = TinyGPT.load(
    CHECKPOINT
)


print(
    "Checkpoint vocabulary:",
    student.vocab_size,
)

print(
    "Tokenizer original vocabulary:",
    OLD_VOCAB_SIZE,
)


# ============================================================
# VERIFY VOCABULARY
# ============================================================

if student.vocab_size != OLD_VOCAB_SIZE:

    raise RuntimeError(
        "\nVocabulary mismatch!\n"
        f"Checkpoint vocabulary: "
        f"{student.vocab_size}\n"
        f"Tokenizer vocabulary: "
        f"{OLD_VOCAB_SIZE}\n\n"
        "The checkpoint and tokenizer must have been "
        "created with the same original vocabulary."
    )


# ============================================================
# RESIZE MODEL
# ============================================================

print("\nResizing model vocabulary...")

student = student.resize_vocab(
    NEW_VOCAB_SIZE
)

print(
    "Model vocabulary:",
    student.vocab_size,
)


# ============================================================
# SAVE TOKENIZER
# ============================================================

TOKENIZER_OUTPUT = (
    "tokenizer_qa"
)

os.makedirs(
    TOKENIZER_OUTPUT,
    exist_ok=True,
)

tokenizer.save_pretrained(
    TOKENIZER_OUTPUT
)

print(
    f"Tokenizer saved to: "
    f"{TOKENIZER_OUTPUT}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Open-Orca...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
)

dataset = dataset.shuffle(
    seed=SEED,
    buffer_size=SHUFFLE_BUFFER,
)

print(
    "Open-Orca streaming dataset ready."
)


# ============================================================
# ENCODE ONE EXAMPLE
# ============================================================

def encode_example(example):
    """
    Convert one Open-Orca example into:

        <|user|>
        QUESTION
        <|assistant|>
        ANSWER
        <|end|>

    The question is truncated to MAX_QUESTION_TOKENS.

    The answer is truncated to MAX_ANSWER_TOKENS.

    The loss mask is:

        0 = ignore
        1 = train

    Therefore the model learns to predict:

        <|assistant|>
        answer
        <|end|>
    """

    question = example.get(
        "question",
        "",
    )

    answer = example.get(
        "response",
        "",
    )

    if question is None:
        question = ""

    if answer is None:
        answer = ""

    question = str(
        question
    ).strip()

    answer = str(
        answer
    ).strip()

    if not question:
        return None

    if not answer:
        return None


    # ========================================================
    # TOKENIZE QUESTION
    # ========================================================

    question_ids = tokenizer.encode(
        question,
        add_special_tokens=False,
    )


    # ========================================================
    # TOKENIZE ANSWER
    # ========================================================

    answer_ids = tokenizer.encode(
        answer,
        add_special_tokens=False,
    )

    if len(answer_ids) == 0:
        return None


    # ========================================================
    # TRUNCATE QUESTION
    # ========================================================

    if len(question_ids) > MAX_QUESTION_TOKENS:

        question_ids = question_ids[
            :MAX_QUESTION_TOKENS
        ]


    # ========================================================
    # TRUNCATE ANSWER
    # ========================================================
    #
    # We ALWAYS keep at least one answer token.
    #
    # Reserve one token for <|end|>.
    #

    answer_ids = answer_ids[
        :MAX_ANSWER_TOKENS
    ]

    if len(answer_ids) == 0:
        return None


    # ========================================================
    # CONSTRUCT SEQUENCE
    # ========================================================

    #
    # USER
    #

    prefix = [
        USER_TOKEN
    ]

    prefix.extend(
        question_ids
    )

    #
    # ASSISTANT
    #

    assistant = [
        ASSISTANT_TOKEN
    ]

    #
    # END
    #

    suffix = [
        END_TOKEN
    ]


    # --------------------------------------------------------
    # Complete token sequence
    # --------------------------------------------------------

    tokens = (
        prefix
        + assistant
        + answer_ids
        + suffix
    )


    # ========================================================
    # LOSS MASK
    # ========================================================
    #
    # We want:
    #
    # <|user|>       IGNORE
    # question       IGNORE
    # <|assistant|>  TRAIN
    # answer         TRAIN
    # <|end|>        TRAIN
    #

    loss_mask = (
        [0] * len(prefix)
        + [1]
        + [1] * len(answer_ids)
        + [1]
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(tokens) > SEQ_LEN + 1:

        raise RuntimeError(
            "Example exceeded context allocation. "
            "Check MAX_QUESTION_TOKENS and "
            "MAX_ANSWER_TOKENS."
        )


    # ========================================================
    # GUARANTEE ANSWER TOKENS
    # ========================================================

    if sum(loss_mask) <= 0:

        return None


    return (
        tokens,
        loss_mask,
    )


# ============================================================
# TRAINING STREAM
# ============================================================

def example_stream():

    for example in dataset:

        result = encode_example(
            example
        )

        if result is None:
            continue

        tokens, loss_mask = result


        # ====================================================
        # NEXT TOKEN PREDICTION
        # ====================================================

        #
        # tokens:
        #
        #   A B C D E
        #
        # input:
        #
        #   A B C D
        #
        # target:
        #
        #   B C D E
        #

        x = tokens[:-1]

        y = tokens[1:]

        target_mask = loss_mask[1:]


        # ====================================================
        # GUARANTEE TRAINING TOKENS
        # ====================================================

        valid_tokens = sum(
            target_mask
        )

        if valid_tokens <= 0:
            continue


        # ====================================================
        # PAD
        # ====================================================

        pad_length = (
            SEQ_LEN
            - len(x)
        )

        if pad_length > 0:

            x = (
                x
                + [0] * pad_length
            )

            y = (
                y
                + [0] * pad_length
            )

            target_mask = (
                target_mask
                + [0] * pad_length
            )


        # ====================================================
        # FINAL SAFETY CHECK
        # ====================================================

        if len(x) != SEQ_LEN:
            continue

        if len(y) != SEQ_LEN:
            continue

        if len(target_mask) != SEQ_LEN:
            continue

        if sum(target_mask) <= 0:
            continue


        yield (
            np.asarray(
                x,
                dtype=np.int64,
            ),

            np.asarray(
                y,
                dtype=np.int64,
            ),

            np.asarray(
                target_mask,
                dtype=np.float32,
            ),
        )


# ============================================================
# BATCH
# ============================================================

def get_batch(
    stream,
    batch_size,
):

    xs = []
    ys = []
    masks = []

    while len(xs) < batch_size:

        x, y, mask = next(
            stream
        )

        xs.append(x)
        ys.append(y)
        masks.append(mask)


    return (
        np.asarray(
            xs,
            dtype=np.int64,
        ),

        np.asarray(
            ys,
            dtype=np.int64,
        ),

        np.asarray(
            masks,
            dtype=np.float32,
        ),
    )


# ============================================================
# MASKED SPARSE CROSS ENTROPY
# ============================================================

def masked_sparse_cross_entropy(
    logits,
    targets,
    mask,
):
    """
    logits:
        [N, VOCAB_SIZE]

    targets:
        [N]

    mask:
        [N]

        0 = ignore
        1 = train
    """

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )


    # ========================================================
    # CHECK
    # ========================================================

    valid_count = mask.sum()

    if float(valid_count) <= 0:

        raise RuntimeError(
            "No valid answer tokens in batch."
        )


    # ========================================================
    # STABLE SOFTMAX
    # ========================================================

    max_logits = logits.data.max(
        axis=-1,
        keepdims=True,
    )

    shifted = (
        logits.data
        - max_logits
    )

    exp_logits = np.exp(
        shifted
    )

    sum_exp = exp_logits.sum(
        axis=-1,
        keepdims=True,
    )

    log_probs = (
        shifted
        - np.log(sum_exp)
    )


    # ========================================================
    # TARGETS
    # ========================================================
    #
    # All targets are valid token IDs.
    #
    # There is NO -100.
    #

    targets = np.clip(
        targets,
        0,
        VOCAB_SIZE - 1,
    )


    # ========================================================
    # TOKEN LOSSES
    # ========================================================

    token_losses = -log_probs[
        np.arange(
            targets.shape[0]
        ),
        targets,
    ]


    # ========================================================
    # MASK
    # ========================================================

    weighted_loss = (
        token_losses
        * mask
    )


    loss_data = (
        weighted_loss.sum()
        / valid_count
    )


    # ========================================================
    # LOSS TENSOR
    # ========================================================

    loss = Euclid.Tensor(
        np.asarray(
            loss_data,
            dtype=np.float32,
        ),
        requires_grad=True,
    )


    # ========================================================
    # BACKWARD
    # ========================================================

    def backward():

        grad = (
            exp_logits
            / sum_exp
        )


        grad[
            np.arange(
                targets.shape[0]
            ),
            targets,
        ] -= 1


        grad *= (
            mask[:, None]
            / valid_count
        )


        logits.grad = grad


    loss._backward = backward

    loss._prev = {
        logits
    }


    return loss


# ============================================================
# OPTIMIZER
# ============================================================

print("\nCreating optimizer...")

optimizer = Euclid.optimizers.Adam(
    learning_rate=LEARNING_RATE,
)

optimizer.set_parameters(
    student.parameters()
)


# ============================================================
# TRAINING
# ============================================================

print(
    "\nStarting Q&A fine-tuning...\n",
    flush=True,
)

stream = example_stream()


# ============================================================
# TRAIN LOOP
# ============================================================

for step in range(
    TRAIN_STEPS
):

    _last_step = step


    # ========================================================
    # BATCH
    # ========================================================

    batch_x, batch_y, batch_mask = get_batch(
        stream,
        BATCH_SIZE,
    )


    # ========================================================
    # FORWARD
    # ========================================================

    x = Euclid.Tensor(
        batch_x
    )

    logits = student(
        x
    )


    # ========================================================
    # FLATTEN
    # ========================================================

    logits_flat = logits.reshape(
        -1,
        VOCAB_SIZE,
    )

    targets_flat = batch_y.reshape(
        -1
    )

    mask_flat = batch_mask.reshape(
        -1
    )


    # ========================================================
    # LOSS
    # ========================================================

    loss = masked_sparse_cross_entropy(
        logits_flat,
        targets_flat,
        mask_flat,
    )


    # ========================================================
    # BACKWARD
    # ========================================================

    optimizer.zero_grad()

    loss.backward()


    # ========================================================
    # UPDATE
    # ========================================================

    optimizer.step()


    # ========================================================
    # LOGGING
    # ========================================================

    valid_tokens = float(
        batch_mask.sum()
    )

    print(
        f"Step {step + 1}/{TRAIN_STEPS} | "
        f"Loss: {float(loss.data):.6f} | "
        f"Answer tokens: {int(valid_tokens)}",
        flush=True,
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        (step + 1) % 1000
        == 0
    ):

        checkpoint_step = (
            step
            + 1
            + STARTING_ITERATION
        )

        model_path = (
            f"tiny_gpt_qa_step_"
            f"{checkpoint_step}.npz"
        )

        tokenizer_path = (
            f"tokenizer_qa_step_"
            f"{checkpoint_step}"
        )


        student.save(
            model_path
        )


        tokenizer.save_pretrained(
            tokenizer_path
        )


        print(
            f"\nSaved model: "
            f"{model_path}",
            flush=True,
        )

        print(
            f"Saved tokenizer: "
            f"{tokenizer_path}\n",
            flush=True,
        )


# ============================================================
# FINISHED
# ============================================================

_training_done = True

print(
    "\nQ&A fine-tuning complete.",
    flush=True,
)