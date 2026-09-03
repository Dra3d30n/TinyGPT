import os
import numpy as np

import euclid as Euclid
from transformers import AutoTokenizer

from tinygpt import TinyGPT


# ============================================================
# CONFIG
# ============================================================
import cupy as cp

free, total = cp.cuda.runtime.memGetInfo()

print(
    f"GPU memory: {free / 1024**3:.2f} GB free / "
    f"{total / 1024**3:.2f} GB total"
)
MODEL_PATH = "tiny_gpt_qa_step_15000.npz"
TOKENIZER_PATH = "tokenizer_qa_step_15000"

SEQ_LEN = 256

TEMPERATURE = 0.8
MAX_NEW_TOKENS = 124

# Set to None for normal sampling.
# Set to 1 for greedy generation.
TOP_K = 50


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH
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

print("Tokenizer size:", len(tokenizer))
print("USER:", USER_TOKEN)
print("ASSISTANT:", ASSISTANT_TOKEN)
print("END:", END_TOKEN)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = TinyGPT.load(
    MODEL_PATH
)

print("Model vocabulary:", model.vocab_size)
print("Model loaded.")


if model.vocab_size != len(tokenizer):
    raise RuntimeError(
        "\nVocabulary mismatch!\n"
        f"Model: {model.vocab_size}\n"
        f"Tokenizer: {len(tokenizer)}\n"
    )


# ============================================================
# SAMPLE TOKEN
# ============================================================
def sample_token(logits):
    """
    Sample one token from logits.
    Converts GPU/CuPy data to CPU/NumPy first.
    """

    if hasattr(logits, "get"):
        logits = logits.get()

    logits = np.asarray(
        logits,
        dtype=np.float32
    )

    logits = logits / TEMPERATURE

    if TOP_K is not None and TOP_K > 0:
        k = min(TOP_K, len(logits))

        top_indices = np.argpartition(
            logits,
            -k
        )[-k:]

        top_logits = logits[top_indices]

        top_logits -= np.max(top_logits)

        probs = np.exp(top_logits)
        probs /= np.sum(probs)

        return int(
            np.random.choice(
                top_indices,
                p=probs
            )
        )

    logits -= np.max(logits)

    probs = np.exp(logits)
    probs /= np.sum(probs)

    return int(
        np.random.choice(
            len(probs),
            p=probs
        )
    )
# ============================================================
# GREEDY TOKEN
# ============================================================

def greedy_token(logits):
    return int(
        np.argmax(
            np.asarray(logits)
        )
    )


# ============================================================
# GENERATE
# ============================================================

def generate(question):

    question = str(question).strip()

    if not question:
        return ""

    # --------------------------------------------------------
    # Encode question
    # --------------------------------------------------------

    question_ids = tokenizer.encode(
        question,
        add_special_tokens=False
    )

    # Leave room for:
    #
    # <|user|>
    # question
    # <|assistant|>
    # answer
    # <|end|>
    #
    # We need at most SEQ_LEN tokens in the model input.

    max_question = (
        SEQ_LEN
        - 2
        - MAX_NEW_TOKENS
    )

    question_ids = question_ids[
        :max_question
    ]

    # --------------------------------------------------------
    # Initial prompt
    # --------------------------------------------------------

    tokens = (
        [USER_TOKEN]
        + question_ids
        + [ASSISTANT_TOKEN]
    )

    generated = []

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for _ in range(MAX_NEW_TOKENS):

        # The model accepts exactly SEQ_LEN tokens
        # in your training setup.

        input_tokens = tokens[-SEQ_LEN:]

        # Pad on the left if necessary.
        #
        # This keeps the most recent context.

        if len(input_tokens) < SEQ_LEN:

            pad_length = (
                SEQ_LEN
                - len(input_tokens)
            )

            input_tokens = (
                [0] * pad_length
                + input_tokens
            )

        x = Euclid.Tensor(
            np.asarray(
                [input_tokens],
                dtype=np.int64
            )
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(x)

        # Expected shape:
        #
        # [batch, sequence, vocab]

        logits_data = Euclid.to_numpy(logits.data)

        next_logits = logits_data[
            0,
            -1,
            :
        ]

        # ----------------------------------------------------
        # Pick token
        # ----------------------------------------------------

        if TOP_K == 1:
            next_token = greedy_token(
                next_logits
            )
        else:
            next_token = sample_token(
                next_logits
            )

        # ----------------------------------------------------
        # End token
        # ----------------------------------------------------

        if next_token == END_TOKEN:
            break

        generated.append(
            next_token
        )

        tokens.append(
            next_token
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# INTERACTIVE TEST
# ============================================================

print("\n" + "=" * 60)
print("TinyGPT Q&A Test")
print("=" * 60)
print("Type a question.")
print("Type 'exit' to quit.")
print("=" * 60)


while True:

    try:
        question = input("\nYou: ")

    except KeyboardInterrupt:
        print("\nExiting.")
        break

    except EOFError:
        print("\nExiting.")
        break

    if question.strip().lower() in {
        "exit",
        "quit",
        "q"
    }:
        print("Exiting.")
        break

    if not question.strip():
        continue

    print("TinyGPT: ", end="", flush=True)

    try:
        answer = generate(
            question
        )

        print(answer)

    except Exception as e:

        print(
            "\nGeneration error:"
        )

        print(
            repr(e)
        )