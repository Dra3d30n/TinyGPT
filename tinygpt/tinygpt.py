import euclid as Euclid
from euclid.backend import xp as np

class TinyGPT(Euclid.networks.Network):

    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model,
        num_heads,
        num_blocks,
        ff_dim,
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ff_dim = ff_dim

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = (
    Euclid.layers.Embedding(
        vocab_size,
        d_model,
        )

    )

        # ----------------------------------------------------
        # Position embedding
        # ----------------------------------------------------

        self.position_embedding = (
            Euclid.layers.Embedding(
                seq_len,
                d_model,
            )
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = [

            Euclid.layers.TransformerBlock(
                d_model,
                num_heads,
                ff_dim,
            )

            for _ in range(num_blocks)
        ]

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.ln_final = (
            Euclid.layers.LayerNorm(
                d_model
            )
        )

        # ----------------------------------------------------
        # Language-model head
        # ----------------------------------------------------

        self.lm_head = (
            Euclid.layers.Dense(
                d_model,
                vocab_size,
            )
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        h = self.token_embedding(x)

        seq_len = x.data.shape[1]

        positions = np.arange(
            seq_len,
            dtype=np.int64,
        )[None, :]

        positions = Euclid.Tensor(
            positions
        )

        p = self.position_embedding(
            positions
        )

        h = h + p

        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)

        return self.lm_head(h)
    # ========================================================
    # PARAMETERS
    # ========================================================

    def parameters(self):

        params = []

        params.extend(
            self.token_embedding.parameters()
        )

        params.extend(
            self.position_embedding.parameters()
        )

        for block in self.blocks:
            params.extend(
                block.parameters()
            )

        params.extend(
            self.ln_final.parameters()
        )

        params.extend(
            self.lm_head.parameters()
        )

        return params


    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path):

        print(
            f"\nSaving TinyGPT to: {path}"
        )

        params = self.parameters()

        save_data = {

            # Architecture
            "vocab_size": np.asarray(
                self.vocab_size
            ),

            "seq_len": np.asarray(
                self.seq_len
            ),

            "d_model": np.asarray(
                self.d_model
            ),

            "num_heads": np.asarray(
                self.num_heads
            ),

            "num_blocks": np.asarray(
                self.num_blocks
            ),

            "ff_dim": np.asarray(
                self.ff_dim
            ),

        }

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        for i, parameter in enumerate(params):

            # Convert CuPy → NumPy if necessary
            data = parameter.data

            if hasattr(data, "get"):
                data = data.get()

            save_data[
                f"param_{i}"
            ] = data

        # ----------------------------------------------------
        # Save everything
        # ----------------------------------------------------

        # Use regular NumPy for file serialization
        import numpy as host_numpy

        host_numpy.savez_compressed(
            path,
            **save_data,
        )

        print(
            f"Saved {len(params)} parameter tensors."
        )


    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(cls, path):

        print(
            f"\nLoading TinyGPT from: {path}"
        )

        import numpy as host_numpy

        data = host_numpy.load(
            path,
            allow_pickle=False,
        )

        # ----------------------------------------------------
        # Architecture
        # ----------------------------------------------------

        vocab_size = int(
            data["vocab_size"]
        )

        seq_len = int(
            data["seq_len"]
        )

        d_model = int(
            data["d_model"]
        )

        num_heads = int(
            data["num_heads"]
        )

        num_blocks = int(
            data["num_blocks"]
        )

        ff_dim = int(
            data["ff_dim"]
        )

        # ----------------------------------------------------
        # Reconstruct model
        # ----------------------------------------------------

        model = cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
        )

        params = model.parameters()

        # ----------------------------------------------------
        # Restore parameters
        # ----------------------------------------------------

        if len(params) == 0:
            raise RuntimeError(
                "Model contains no parameters."
            )

        for i, parameter in enumerate(params):

            key = f"param_{i}"

            if key not in data:
                raise RuntimeError(
                    f"Missing parameter: {key}"
                )

            saved = data[key]

            # Move NumPy → Euclid backend
            parameter.data[...] = Euclid.xp.asarray(
                saved,
                dtype=parameter.data.dtype,
            )

        print(
            f"Loaded {len(params)} parameter tensors."
        )

        return model

    # ========================================================
    # RESIZE VOCABULARY
    # ========================================================

    def resize_vocab(self, new_vocab_size):

        old_vocab_size = self.vocab_size

        if new_vocab_size <= old_vocab_size:
            raise ValueError(
                f"New vocabulary size ({new_vocab_size}) "
                f"must be greater than old vocabulary size "
                f"({old_vocab_size})."
            )

        print(
            f"\nResizing vocabulary: "
            f"{old_vocab_size} -> {new_vocab_size}"
        )

        # ----------------------------------------------------
        # Create a new model with the larger vocabulary
        # ----------------------------------------------------

        new_model = TinyGPT(
            vocab_size=new_vocab_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            ff_dim=self.ff_dim,
        )

        old_params = self.parameters()
        new_params = new_model.parameters()

        if len(old_params) != len(new_params):
            raise RuntimeError(
                "Parameter count changed while resizing vocabulary."
            )

        # ----------------------------------------------------
        # Copy weights
        # ----------------------------------------------------

        resized = []

        for i, (old, new) in enumerate(
            zip(old_params, new_params)
        ):

            old_shape = old.data.shape
            new_shape = new.data.shape

            if old_shape == new_shape:

                # Normal parameter
                new.data[...] = old.data

            else:

                # ------------------------------------------------
                # Vocabulary-dependent parameter
                #
                # Embedding:
                #   [old_vocab, d_model]
                #       ->
                #   [new_vocab, d_model]
                #
                # LM head:
                #   [d_model, old_vocab]
                #       ->
                #   [d_model, new_vocab]
                #
                # LM bias:
                #   [old_vocab]
                #       ->
                #   [new_vocab]
                # ------------------------------------------------

                if len(old_shape) != len(new_shape):
                    raise RuntimeError(
                        f"Unexpected shape change for parameter {i}: "
                        f"{old_shape} -> {new_shape}"
                    )

                # Start by keeping the new parameter's
                # random initialization.

                if len(old_shape) == 1:

                    # Bias: [vocab]
                    if (
                        old_shape[0] == old_vocab_size
                        and new_shape[0] == new_vocab_size
                    ):
                        new.data[:old_vocab_size] = old.data
                        resized.append(i)

                    else:
                        raise RuntimeError(
                            f"Unexpected 1D shape change: "
                            f"{old_shape} -> {new_shape}"
                        )

                elif len(old_shape) == 2:

                    # Case 1: Embedding
                    #
                    # [vocab, d_model]
                    if (
                        old_shape[0] == old_vocab_size
                        and new_shape[0] == new_vocab_size
                        and old_shape[1] == new_shape[1]
                    ):

                        new.data[:old_vocab_size, :] = old.data
                        resized.append(i)

                    # Case 2: LM head
                    #
                    # [d_model, vocab]
                    elif (
                        old_shape[1] == old_vocab_size
                        and new_shape[1] == new_vocab_size
                        and old_shape[0] == new_shape[0]
                    ):

                        new.data[:, :old_vocab_size] = old.data
                        resized.append(i)

                    else:
                        raise RuntimeError(
                            f"Unexpected 2D shape change: "
                            f"{old_shape} -> {new_shape}"
                        )

                else:

                    raise RuntimeError(
                        f"Unexpected parameter shape change: "
                        f"{old_shape} -> {new_shape}"
                    )

        # ----------------------------------------------------
        # Update vocabulary metadata
        # ----------------------------------------------------

        new_model.vocab_size = new_vocab_size

        print(
            f"Vocabulary resized successfully."
        )

        print(
            f"Vocabulary-dependent tensors resized: "
            f"{resized}"
        )

        print(
            "New token weights were randomly initialized."
        )

        return new_model