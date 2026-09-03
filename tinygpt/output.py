
import random
import threading
import tkinter as tk
from tkinter import scrolledtext

import torch
import numpy as host_np

import euclid as Euclid
import cupy as np

from transformers import AutoTokenizer
from tinygpt import TinyGPT


# ============================================================
# CONFIG
# ============================================================

TINYGPT_PATH = "models/tiny_gpt_distilled_step_220000.npz"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.5
REPETITION_PENALTY = 1.35


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading TinyGPT...")
Euclid.change_device("gpu")
Euclid.change_precision(np.float32)
tiny_tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

tiny_model = TinyGPT.load(
    TINYGPT_PATH
)

tiny_model.eval()

print("TinyGPT loaded.")

print(
    "Parameters:",
    sum(
        p.data.size
        for p in tiny_model.parameters()
    )
)


# ============================================================
# SEED
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
# GENERATION
# ============================================================

def generate(prompt, seed=None):

    if seed is not None:
        set_seed(seed)

    prompt_ids = tiny_tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    token_ids = prompt_ids.copy()

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

        # Last position
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

        logits -= np.max(logits)

        probs = np.exp(logits)
        probs /= np.sum(probs)

        # Greedy decoding
        next_token = int(
            np.argmax(logits)
        )

        token_ids.append(next_token)

        # EOS
        if (
            tiny_tokenizer.eos_token_id is not None
            and next_token == tiny_tokenizer.eos_token_id
        ):
            break

    # --------------------------------------------------------
    # Decode generated tokens
    # --------------------------------------------------------

    generated_ids = token_ids[
        len(prompt_ids):
    ]

    response = tiny_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # --------------------------------------------------------
    # Cut at the most recent period
    # --------------------------------------------------------

    last_period = response.rfind(".")

    if last_period != -1:
        response = response[
            :last_period + 1
        ]

    return response.strip()


# ============================================================
# GUI
# ============================================================

class TinyGPTApp:

    def __init__(self, root):

        self.root = root
        self.root.title("TinyGPT")
        self.root.geometry("800x600")
        self.root.minsize(500, 400)

        self.generating = False

        # ----------------------------------------------------
        # Main frame
        # ----------------------------------------------------

        main = tk.Frame(
            root,
            bg="#1e1e1e"
        )

        main.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = tk.Frame(
            main,
            bg="#252526",
            height=60
        )

        header.pack(
            fill=tk.X
        )

        title = tk.Label(
            header,
            text="TinyGPT",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#252526"
        )

        title.pack(
            side=tk.LEFT,
            padx=20,
            pady=15
        )

        status = tk.Label(
            header,
            text="Ready",
            font=("Arial", 10),
            fg="#aaaaaa",
            bg="#252526"
        )

        status.pack(
            side=tk.RIGHT,
            padx=20
        )

        self.status = status

        # ----------------------------------------------------
        # Chat
        # ----------------------------------------------------

        self.chat = scrolledtext.ScrolledText(
            main,
            wrap=tk.WORD,
            font=("Arial", 12),
            bg="#1e1e1e",
            fg="#eeeeee",
            insertbackground="white",
            borderwidth=0,
            padx=15,
            pady=15,
            state=tk.DISABLED
        )

        self.chat.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )

        # Text tags
        self.chat.tag_config(
            "user",
            foreground="#6cb6ff",
            font=("Arial", 12, "bold")
        )

        self.chat.tag_config(
            "bot",
            foreground="#8fd694",
            font=("Arial", 12, "bold")
        )

        self.chat.tag_config(
            "message",
            foreground="#eeeeee",
            font=("Arial", 12)
        )

        # ----------------------------------------------------
        # Input area
        # ----------------------------------------------------

        bottom = tk.Frame(
            main,
            bg="#252526"
        )

        bottom.pack(
            fill=tk.X,
            padx=10,
            pady=(0, 10)
        )

        self.input_box = tk.Entry(
            bottom,
            font=("Arial", 13),
            bg="#333333",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT
        )

        self.input_box.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(10, 5),
            pady=10,
            ipady=8
        )

        self.input_box.bind(
            "<Return>",
            self.send_message
        )

        self.send_button = tk.Button(
            bottom,
            text="Send",
            font=("Arial", 11, "bold"),
            bg="#4c8bf5",
            fg="white",
            activebackground="#3b78dc",
            activeforeground="white",
            relief=tk.FLAT,
            command=self.send_message
        )

        self.send_button.pack(
            side=tk.RIGHT,
            padx=(5, 10),
            pady=10,
            ipadx=15,
            ipady=6
        )

        # Focus input
        self.input_box.focus_set()



    # ========================================================
    # CHAT DISPLAY
    # ========================================================

    def add_message(
        self,
        name,
        message,
        tag
    ):

        self.chat.config(
            state=tk.NORMAL
        )

        self.chat.insert(
            tk.END,
            f"{name}: ",
            tag
        )

        self.chat.insert(
            tk.END,
            f"{message}\n\n",
            "message"
        )

        self.chat.see(
            tk.END
        )

        self.chat.config(
            state=tk.DISABLED
        )

    # ========================================================
    # SEND
    # ========================================================

    def send_message(self, event=None):

        if self.generating:
            return

        prompt = self.input_box.get().strip()

        if not prompt:
            return

        self.input_box.delete(
            0,
            tk.END
        )

        self.add_message(
            "You",
            prompt,
            "user"
        )

        self.generating = True

        self.send_button.config(
            state=tk.DISABLED
        )

        self.input_box.config(
            state=tk.DISABLED
        )

        self.status.config(
            text="Generating..."
        )

        # Run model in background
        thread = threading.Thread(
            target=self.generate_response,
            args=(prompt,),
            daemon=True
        )

        thread.start()

    # ========================================================
    # GENERATE
    # ========================================================

    def generate_response(
        self,
        prompt
    ):

        try:

            response = generate(
                prompt,
                seed=random.randint(
                    0,
                    2**32 - 1
                )
            )

            if not response:
                response = "I don't know."

        except Exception as e:

            response = (
                f"Generation error: {e}"
            )

        # Update GUI on main thread
        self.root.after(
            0,
            lambda: self.finish_response(
                response
            )
        )

    # ========================================================
    # FINISH
    # ========================================================

    def finish_response(
        self,
        response
    ):

        self.add_message(
            "TinyGPT",
            response,
            "bot"
        )

        self.generating = False

        self.send_button.config(
            state=tk.NORMAL
        )

        self.input_box.config(
            state=tk.NORMAL
        )

        self.status.config(
            text="Ready"
        )

        self.input_box.focus_set()


# ============================================================
# START GUI
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = TinyGPTApp(
        root
    )

    root.mainloop()

