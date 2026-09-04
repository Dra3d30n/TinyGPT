
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

        self.token_embedding = Euclid.layers.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = Euclid.layers.Embedding(
            seq_len,
            d_model,
        )

        self.blocks = [
            Euclid.layers.TransformerBlock(
                d_model,
                num_heads,
                ff_dim,
            )
            for _ in range(num_blocks)
        ]

        self.ln_final = Euclid.layers.LayerNorm(
            d_model
        )

        self.lm_head = Euclid.layers.Dense(
            d_model,
            vocab_size,
        )

    def forward(self, x):
        h = self.token_embedding(x)

        seq_len = x.data.shape[1]

        positions = np.arange(
            seq_len,
            dtype=np.int64,
        )[None, :]

        positions = Euclid.Tensor(positions)

        p = self.position_embedding(positions)

        h = h + p

        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)

        return self.lm_head(h)

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

    def save(self, path):
        print(f"\nSaving TinyGPT to: {path}")

        params = self.parameters()

        save_data = {
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

        for i, parameter in enumerate(params):
            data = parameter.data

            if hasattr(data, "get"):
                data = data.get()

            save_data[f"param_{i}"] = data

        import numpy as host_numpy

        host_numpy.savez_compressed(
            path,
            **save_data,
        )

        print(
            f"Saved {len(params)} parameter tensors."
        )

    @classmethod
    def load(cls, path):
        print(f"\nLoading TinyGPT from: {path}")

        import numpy as host_numpy

        data = host_numpy.load(
            path,
            allow_pickle=False,
        )

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

        model = cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
        )

        params = model.parameters()

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

            parameter.data[...] = Euclid.xp.asarray(
                saved,
                dtype=parameter.data.dtype,
            )

        print(
            f"Loaded {len(params)} parameter tensors."
        )

        return model

