import euclid as Euclid
import archid as Archid
from tinygpt import TinyGPT
from euclid.backend import xp

import numpy as host_np
from datasets import load_dataset
from transformers import AutoTokenizer

TEACHER_MODEL = "mistralai/Mistral-7B-v0.1"

EPOCHS = 100

D_MODEL = 768
NUM_HEADS = 12
NUM_BLOCKS = 12
FF_DIM = 3072

SEQ_LEN = 512
BATCH_SIZE = 2

tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL
)

VOCAB_SIZE = tokenizer.vocab_size

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
).shuffle(
    seed=9,
    buffer_size=10000,
)

dataset_iter = iter(dataset)


def generate_batch(batch_size):

    tokens = []

    while len(tokens) < batch_size * SEQ_LEN + 1:

        example = next(dataset_iter)

        text = example["text"]

        if not text or len(text.strip()) < 20:
            continue

        new_tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        tokens.extend(new_tokens)

    tokens = tokens[
        :batch_size * SEQ_LEN + 1
    ]

    tokens = host_np.asarray(
        tokens,
        dtype=host_np.int64,
    )

    x = tokens[:-1].reshape(
        batch_size,
        SEQ_LEN,
    )

    y = tokens[1:].reshape(
        batch_size,
        SEQ_LEN,
    )

    return x, y


network = TinyGPT(
    VOCAB_SIZE,
    SEQ_LEN,
    D_MODEL,
    NUM_HEADS,
    NUM_BLOCKS,
    FF_DIM,
)


softmax = Euclid.layers.activations.Softmax()


class CrossEntropy(Euclid.losses.CrossEntropy):

    def forward(self, prediction, target):

        # prediction:
        # (B, S, V)
        #
        # target:
        # (B, S)

        B, S, V = prediction.data.shape

        # ----------------------------------------------------
        # Flatten batch + sequence
        # ----------------------------------------------------

        prediction = prediction.reshape(
            B * S,
            V,
        )

        target = target.reshape(
            B * S,
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        probabilities = softmax.forward(
            prediction
        )

        # ----------------------------------------------------
        # Gather probability of correct token
        # ----------------------------------------------------

        target_indices = target.data.astype(
            xp.int64
        ).reshape(
            B * S,
            1,
        )

        correct_probabilities = probabilities.gather(
            axis=1,
            indices=target_indices,
        )

        # ----------------------------------------------------
        # Cross entropy
        # ----------------------------------------------------

        loss = -correct_probabilities.log().mean()

        return loss


data = Euclid.data.StreamDataLoader(
    generate_fn=generate_batch,
    batch_size=BATCH_SIZE,
    steps_per_epoch=1,
)

learning = Archid.learning.supervised.SupervisedLearning(
    CrossEntropy(),
    Euclid.optimizers.Adam(
        learning_rate=1e-4
    )
)

model = Archid.model.Model(
    network,
    learning,
)

print("training start")

model.train(
    data,
    EPOCHS,
)

print("training stop")