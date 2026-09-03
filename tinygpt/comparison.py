import random
import json
import torch
import numpy as host_np

import euclid as Euclid
from euclid.backend import xp as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from tinygpt import TinyGPT


# ============================================================
# CONFIG
# ============================================================

GPT1_NAME = "openai-community/openai-gpt"

TINYGPT_PATH = "models/tiny_gpt_distilled_step_250000.npz"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.3
REPETITION_PENALTY = 1.35

SEEDS = [42, 123]

OUTPUT_FILE = "tinygpt_vs_gpt1_results.json"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD GPT-1
# ============================================================

print("\nLoading GPT-1...")

gpt1_tokenizer = AutoTokenizer.from_pretrained(
    GPT1_NAME
)

gpt1_model = AutoModelForCausalLM.from_pretrained(
    GPT1_NAME
)

gpt1_model.to(device)
gpt1_model.eval()

print("GPT-1 loaded.")


# ============================================================
# LOAD TINYGPT
# ============================================================

print("\nLoading TinyGPT 200k...")

tiny_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

tiny_model = TinyGPT.load(
    TINYGPT_PATH
)

print("TinyGPT loaded.")

print(
    "TinyGPT parameters:",
    sum(
        p.data.size
        for p in tiny_model.parameters()
    )
)


# ============================================================
# PROMPTS
# ============================================================

prompts = [

    """The development of modern computers did not happen because of a single invention. Instead, it was the result of many discoveries building upon one another over several centuries. Early mathematicians developed systems for representing numbers, while inventors created mechanical devices capable of performing calculations. The invention of electronic components eventually made it possible to construct machines that could perform thousands of operations in a fraction of a second. One of the most important ideas in this history was the realization that a machine could store not only data, but also the instructions needed to manipulate that data. This idea changed the nature of computing because""",


    """A neural network begins with a surprisingly simple idea. Instead of writing an explicit rule for every possible situation, we can create a mathematical system containing many adjustable parameters and allow those parameters to change as the system is exposed to examples. At first, the network knows almost nothing about the task it is supposed to perform. Its parameters contain essentially random values, and its predictions are therefore poor. However, after seeing an example, the network can compare its prediction with the correct answer and determine how its parameters should change. Repeating this process many times eventually allows""",


    """When a CPU executes a program, it does not understand the source code in the same way that a programmer does. A processor sees a sequence of binary instructions and data stored in memory. Each instruction tells the processor to perform some operation, such as adding two numbers, loading a value from memory, storing a result, or comparing two values. The processor repeatedly retrieves instructions and determines what operation they represent. This process is known as the instruction cycle, and it can be divided into several stages. First,""",


    """For much of human history, observing the natural world was inseparable from trying to explain it. People noticed that objects fell toward the ground, that the seasons followed recurring patterns, and that the positions of stars changed in predictable ways. However, observation by itself could not always distinguish between competing explanations. A person might observe the same phenomenon as another person but reach a completely different conclusion about its cause. The development of modern science introduced a more systematic approach in which scientists began to formulate hypotheses and test them against evidence. This changed scientific investigation because""",


    """The first time a student writes a program, the instructions may appear almost magical. A few lines of text are typed into an editor, and after pressing a button, the computer performs exactly the operations described by those lines. In reality, however, the computer does not directly understand the programming language used by the student. The source code must pass through several stages before the processor can execute it. Characters are first grouped into meaningful tokens, those tokens are analyzed according to the grammar of the language, and the resulting structure can then be transformed into a representation that the machine can execute. This process begins when""",


    """Consider what happens when a person looks at a photograph of a handwritten number. Almost immediately, the person can recognize whether the image contains a three, a seven, or some other digit. A computer, however, does not naturally see the photograph as a collection of meaningful shapes. It initially receives a grid of numerical pixel values. The challenge is therefore to find a way of transforming those numbers into a representation from which the identity of the digit can be determined. One approach is to construct a neural network and train it using thousands of labeled examples. During training,""",


    """Electricity is often described using words such as voltage, current, and resistance, but these terms can be difficult to understand without connecting them to a physical system. Imagine a simple circuit containing a battery, two wires, and a light bulb. When the circuit is incomplete, the bulb remains dark even though the battery has a voltage across its terminals. When the wires are connected to form a complete path, something changes: electrical current can flow through the circuit. The battery provides a difference in electrical potential, while the components in the circuit determine how that energy is transferred. In the case of the light bulb,""",


    """The idea of artificial intelligence has existed for much longer than modern neural networks. Early researchers imagined machines that could manipulate symbols, solve logical problems, and eventually perform tasks that appeared to require human intelligence. Some of these approaches were successful in carefully controlled environments, but they often struggled when confronted with the enormous variety found in the real world. A system might be able to follow a collection of carefully written rules, yet fail when the input differed slightly from the situations anticipated by its designers. Machine learning introduced a different approach. Rather than attempting to describe every possible rule explicitly, researchers began to ask whether a machine could instead learn useful patterns directly from examples. This led to""",


    """Language is an unusual kind of data because its meaning depends heavily on context. The word “bank,” for example, can refer to a financial institution or the side of a river, and the surrounding words usually provide the information necessary to determine which meaning is intended. A computer processing text therefore needs more than a simple dictionary of definitions. It must somehow represent relationships between words and account for the words that appear around them. Early language models attempted to do this by considering relatively small windows of text, but this created a fundamental limitation. As the distance between related words increased, it became increasingly difficult for the model to preserve the relevant information. The development of attention mechanisms changed this by allowing""",


    """A scientist standing at the edge of a forest notices that several groups of trees appear to be growing at different rates. Some receive direct sunlight for most of the day, while others are shaded by taller trees. It would be tempting to immediately conclude that sunlight is responsible for the difference, but several other explanations are possible. Perhaps the soil contains different amounts of nutrients, perhaps one group receives more water, or perhaps the trees belong to different species. To determine which explanation is most likely, the scientist must design an experiment that separates these possible causes. The first step is to""",
]


# ============================================================
# SEEDING
# ============================================================

def set_seed(seed):
    random.seed(seed)
    host_np.random.seed(seed)

    try:
        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GPT-1 GENERATION
# ============================================================

def generate_gpt1(prompt, seed):

    set_seed(seed)

    inputs = gpt1_tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():

        output = gpt1_model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            temperature=0.8,
            top_p=0.9,

            do_sample=True,

            pad_token_id=
                gpt1_tokenizer.eos_token_id,
        )

    input_length = inputs[
        "input_ids"
    ].shape[1]

    generated = gpt1_tokenizer.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    )

    return generated.strip()


# ============================================================
# TINYGPT GENERATION
# ============================================================

def generate_tinygpt(
    prompt,
    seed,
):

    set_seed(seed)

    token_ids = tiny_tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    if len(token_ids) == 0:
        raise ValueError(
            "Prompt produced no tokens."
        )

    for _ in range(MAX_NEW_TOKENS):

        context = token_ids[
            -tiny_model.seq_len:
        ]

        x = np.asarray(
            [context],
            dtype=np.int64,
        )

        x = Euclid.Tensor(x)

        # Forward pass
        logits = tiny_model(x)

        # Final token
        logits_data = logits.data[
            :, -1, :
        ]

        # ----------------------------------------------------
        # Repetition penalty
        # ----------------------------------------------------

        for token_id in set(token_ids):

            score = logits_data[
                0,
                token_id
            ]

            if score > 0:
                logits_data[
                    0,
                    token_id
                ] /= REPETITION_PENALTY

            else:
                logits_data[
                    0,
                    token_id
                ] *= REPETITION_PENALTY

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        logits = (
            logits_data[0]
            / TEMPERATURE
        )

        # Stable softmax
        logits = (
            logits
            - np.max(logits)
        )

        probs = np.exp(logits)

        probs /= np.sum(probs)

        # Sample
        next_token = int(
            np.random.choice(
                len(probs),
                size=1,
                p=probs,
            )[0]
        )

        token_ids.append(
            next_token
        )

        # EOS
        if (
            tiny_tokenizer.eos_token_id
            is not None
            and next_token
            == tiny_tokenizer.eos_token_id
        ):
            break

    # Return ONLY generated text
    prompt_token_count = len(
        tiny_tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
    )

    generated_ids = token_ids[
        prompt_token_count:
    ]

    return tiny_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# RESULTS
# ============================================================

wins = {
    "GPT-1": 0,
    "TinyGPT": 0,
    "Tie": 0,
}

results = []


# ============================================================
# BLIND EVALUATION
# ============================================================

total_trials = (
    len(prompts)
    * len(SEEDS)
)

trial = 0


for prompt_index, prompt in enumerate(prompts):

    for seed in SEEDS:

        trial += 1

        print("\n")
        print("=" * 80)
        print(
            f"TRIAL {trial}/{total_trials}"
        )
        print("=" * 80)

        print("\nPROMPT:")
        print(prompt)

        print(
            f"\nSeed: {seed}"
        )

        print("\nGenerating GPT-1...")

        gpt1_output = generate_gpt1(
            prompt,
            seed,
        )

        print("Generating TinyGPT...")

        tiny_output = generate_tinygpt(
            prompt,
            seed,
        )

        # ----------------------------------------------------
        # Randomize A/B
        # ----------------------------------------------------

        if random.choice([True, False]):

            a_model = "GPT-1"
            b_model = "TinyGPT"

            a_output = gpt1_output
            b_output = tiny_output

        else:

            a_model = "TinyGPT"
            b_model = "GPT-1"

            a_output = tiny_output
            b_output = gpt1_output

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\n")
        print("-" * 80)
        print("OUTPUT A")
        print("-" * 80)

        print(a_output)

        print("\n")
        print("-" * 80)
        print("OUTPUT B")
        print("-" * 80)

        print(b_output)

        print("\n")
        print("=" * 80)
        print("WHICH OUTPUT IS BETTER?")
        print("=" * 80)

        print("A = Output A")
        print("B = Output B")
        print("T = Tie")

        while True:

            choice = input(
                "\nYour choice: "
            ).strip().upper()

            if choice in (
                "A",
                "B",
                "T",
            ):
                break

            print(
                "Please enter A, B, or T."
            )

        # ----------------------------------------------------
        # Determine winner
        # ----------------------------------------------------

        if choice == "A":

            winner = a_model

        elif choice == "B":

            winner = b_model

        else:

            winner = "Tie"

        wins[winner] += 1

        # ----------------------------------------------------
        # Save trial
        # ----------------------------------------------------

        results.append({

            "trial": trial,

            "prompt_index":
                prompt_index,

            "prompt":
                prompt,

            "seed":
                seed,

            # Hidden from user during evaluation
            "a_model":
                a_model,

            "b_model":
                b_model,

            "a_output":
                a_output,

            "b_output":
                b_output,

            "choice":
                choice,

            "winner":
                winner,
        })

        print(
            f"\nRecorded winner: {winner}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    f"GPT-1:   {wins['GPT-1']}"
)

print(
    f"TinyGPT: {wins['TinyGPT']}"
)

print(
    f"Ties:    {wins['Tie']}"
)

print(
    f"Total:   {total_trials}"
)

print("\nPercentages:")

for model in (
    "GPT-1",
    "TinyGPT",
    "Tie",
):

    percentage = (
        wins[model]
        / total_trials
        * 100
    )

    print(
        f"{model:8}: "
        f"{percentage:.1f}%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "models": {
                "gpt1":
                    GPT1_NAME,

                "tinygpt":
                    TINYGPT_PATH,
            },

            "prompts":
                prompts,

            "seeds":
                SEEDS,

            "max_new_tokens":
                MAX_NEW_TOKENS,

            "summary":
                wins,

            "total_trials":
                total_trials,

            "results":
                results,
        },

        f,

        indent=2,

        ensure_ascii=False,
    )


print(
    f"\nDetailed results saved to "
    f"{OUTPUT_FILE}"
)
