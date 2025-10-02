import torch.nn as nn


class CrossAttAdaLN(nn.Module):
    """Wraps a cross-attention module for the Perceiver:
    - Modulates the multi-source embeddings that will serve as keys and values
    - Optionally modulates the latent array Lv using the conditioning information Ld.
    - Modulates the output Lv using skip connections and gating mechanisms.
    - Normalizes the latent arrays Lc and Ld before passing them to the wrapped module,
      and applies a residual connection to them.
    """

    def __init__(self, module, values_dim, coords_dim, modulate_latents=False):
        """
        Args:
            module (nn.Module): The module to wrap.
            values_dim (int): Embedding dimension for the values of each source.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            modulate_latents (bool): Whether to modulate Lv
                using the conditioning information Ld before passing it to the wrapped module.
        """
        super().__init__()
        self.module = module
        self.modulate_latents = modulate_latents

        # Normalization and conditioning for the multi-source values.
        self.values_norm = nn.LayerNorm(values_dim)
        self.values_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, values_dim * 2))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.values_cond_proj[1].weight)
        nn.init.zeros_(self.values_cond_proj[1].bias)

        # Normalization and conditioning for the latent values array Lv.
        self.latents_norm = nn.LayerNorm(values_dim)
        if modulate_latents:
            # Scale, shift, gate
            self.latents_cond_proj = nn.Sequential(
                nn.SiLU(), nn.Linear(values_dim, values_dim * 3)
            )
        else:
            # If not modulating Lv before feeding it to the wrapped module,
            # only produce a gate for the output modulation.
            self.latents_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, values_dim))
        nn.init.zeros_(self.latents_cond_proj[1].weight)
        nn.init.zeros_(self.latents_cond_proj[1].bias)

        # Normalization for Lc
        self.coords_norm = nn.LayerNorm(coords_dim)

    def forward(self, x, Lv, Lc, Ld):
        """Args:
            x (dict): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys
                "values", "coords", and "conditioning".
            Lv (tensor): Latent values array of shape (latent_seq_len, values_dim).
            Lc (tensor): Latent coordinates array of shape (latent_seq_len, coords_dim).
            Ld (tensor): Latent conditioning array of shape (latent_seq_len, values_dim).
        Returns:
            Lv (tensor): Updated latent values array of shape (latent_seq_len, values_dim).
            Lc (tensor): Updated latent coordinates array of shape (latent_seq_len, coords_dim).
            Ld (tensor): Updated latent conditioning array of shape (latent_seq_len, values_dim).
        """
        modulated_data = {}

        # Modulate the multi-source values
        for source_index_pair, source_data in x.items():
            # Create a new dict to avoid modifying the input one in-place
            modulated_data[source_index_pair] = {k: v for k, v in source_data.items()}

            # Process values
            values_x = self.values_norm(source_data["values"])

            # Apply conditioning if available
            cond = source_data.get("conditioning", None)
            if cond is not None:
                # Apply conditioning to values
                values_shift, values_scale = self.values_cond_proj(cond).chunk(2, dim=-1)
                values_x = values_x * (values_scale + 1) + values_shift

            # Save the module's inputs for that source
            modulated_data[source_index_pair]["values"] = values_x

        # Modulate the latent values Lv using the conditioning Ld
        Lv_in = self.latents_norm(Lv)
        ld_proj = self.latents_cond_proj(Ld)
        if self.modulate_latents:
            lv_scale, lv_shift, lv_gate = ld_proj.chunk(3, dim=-1)
            Lv_in = Lv_in * (lv_scale + 1) + lv_shift
        else:
            lv_gate = ld_proj

        # Normalize Lc
        Lc_in = self.coords_norm(Lc)

        # Apply the wrapped module with the modulated inputs
        Lv_out, Lc_out, Ld_out = self.module(modulated_data, Lv_in, Lc_in)

        # Modulate the output Lv using skip connections and gating mechanisms
        Lv_out = Lv + Lv_out * lv_gate
        Lc_out = Lc + Lc_out  # Simpler residual connection for Lc and Ld
        Ld_out = Ld + Ld_out

        return Lv_out, Lc_out, Ld_out


class DecoderAdaLN(nn.Module):
    """Adaptive LayerNorm (AdaLN) for the Perceiver decoder:
    - Modulates the multi-source embeddings that will serve as queries.
    - Modulates Lv using the conditioning information Ld.
    - Normalizes Lc before passing it to the wrapped module.
    - Modulates the output multi-source values using skip connections and gating mechanisms.
    """

    def __init__(self, module, values_dim, coords_dim):
        """
        Args:
            module (nn.Module): The module to wrap.
            values_dim (int): Embedding dimension for the values of each source.
            coords_dim (int): Embedding dimension for the coordinates of each source.
        """
        super().__init__()
        self.module = module

        # Normalization and conditioning for the latent values array Lv.
        self.latents_norm = nn.LayerNorm(values_dim)
        self.latents_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, values_dim * 2))
        nn.init.zeros_(self.latents_cond_proj[1].weight)
        nn.init.zeros_(self.latents_cond_proj[1].bias)

        # Normalization and conditioning for the multi-source values.
        self.values_norm = nn.LayerNorm(values_dim)
        self.values_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, values_dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.values_cond_proj[1].weight)
        nn.init.zeros_(self.values_cond_proj[1].bias)

        # Normalization for Lc
        self.coords_norm = nn.LayerNorm(coords_dim)

    def forward(self, x, Lv, Lc, Ld):
        """Args:
            x (dict): Dictionary of inputs, such that
                x[(source_name, index)] contains the keys
                "values", "coords", and "conditioning".
            Lv (tensor): Latent values array of shape (latent_seq_len, values_dim).
            Lc (tensor): Latent coordinates array of shape (latent_seq_len, coords_dim).
            Ld (tensor): Latent conditioning array of shape (latent_seq_len, values_dim).
        Returns:
            y (dict): Dictionary of outputs, such that
                y[(source_name, index)] contains the key "values"
                with the same shape as x[(source_name, index)]["values"].
        """

        # Modulate the latent values Lv using the conditioning Ld
        Lv = self.latents_norm(Lv)
        ld_proj = self.latents_cond_proj(Ld)
        lv_scale, lv_shift = ld_proj.chunk(2, dim=-1)
        Lv = Lv * (lv_scale + 1) + lv_shift
        # Normalize Lc
        Lc = self.coords_norm(Lc)

        # Modulate the multi-source values
        modulated_data = {}
        values_skips, values_gates = {}, {}
        for source_index_pair, source_data in x.items():
            # Create a new dict to avoid modifying the input one in-place
            modulated_data[source_index_pair] = {k: v for k, v in source_data.items()}

            # Normalize and condition values
            values_x = self.values_norm(source_data["values"])
            cond = source_data.get("conditioning", None)
            if cond is not None:
                # Apply conditioning to values
                v_shift, v_scale, v_gate = self.values_cond_proj(cond).chunk(3, dim=-1)
                x_in = values_x * (v_scale + 1) + v_shift

            # Save the module's input, skip and gate for that source
            modulated_data[source_index_pair]["values"] = x_in
            values_skips[source_index_pair] = values_x
            values_gates[source_index_pair] = v_gate

        # Apply the wrapped module with the modulated inputs
        y = self.module(modulated_data, Lv, Lc)
        # Modulate the output y using skip connections and gating mechanisms
        for source_index_pair, y_data in y.items():
            y[source_index_pair] = (
                values_skips[source_index_pair] + y_data * values_gates[source_index_pair]
            )
        return y


class LatentAdaLN(nn.Module):
    """Adaptive LayerNorm (AdaLN) for modules that only work on the latent arrays.
    - Modulates the latent array Lv using the conditioning information Ld.
    - Modulates the output Lv using skip connections and gating mechanisms.
    - Normalizes Lc before passing it to the wrapped module,
        and applies a residual connection to it.
    """

    def __init__(self, module, values_dim, coords_dim):
        """
        Args:
            module (nn.Module): The module to wrap.
            coords_dim (int): Embedding dimension for the coordinates of each source.
            values_dim (int): Embedding dimension for the values of each source.
        """
        super().__init__()
        self.module = module

        # Normalization and conditioning for the latent values array Lv.
        self.latents_norm = nn.LayerNorm(values_dim)
        # Scale, shift, gate
        self.latents_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(values_dim, values_dim * 3))
        # Initialize the weights of the conditional normalization to zero (no effect)
        nn.init.zeros_(self.latents_cond_proj[1].weight)
        nn.init.zeros_(self.latents_cond_proj[1].bias)

        # Normalization for Lc
        self.coords_norm = nn.LayerNorm(coords_dim)

    def forward(self, Lv, Lc, Ld):
        """Args:
            Lv (tensor): Latent values array of shape (latent_seq_len, values_dim).
            Lc (tensor): Latent coordinates array of shape (latent_seq_len, coords_dim).
            Ld (tensor): Latent conditioning array of shape (latent_seq_len, values_dim).
        Returns:
            Lv (tensor): Updated latent values array of shape (latent_seq_len, values_dim).
            Lc (tensor): Updated latent coordinates array of shape (latent_seq_len, coords_dim).
            Ld (tensor): Updated latent conditioning array of shape (latent_seq_len, values_dim).
        """
        # Modulate the latent values Lv using the conditioning Ld
        Lv_in = self.latents_norm(Lv)
        lv_scale, lv_shift, lv_gate = self.latents_cond_proj(Ld).chunk(3, dim=-1)
        Lv_in = Lv_in * (lv_scale + 1) + lv_shift

        # Normalize Lc
        Lc_in = self.coords_norm(Lc)

        # Apply the wrapped module with the modulated inputs
        Lv_out, Lc_out, Ld_out = self.module(Lv_in, Lc_in, Ld)

        # Modulate the output Lv using skip connections and gating mechanisms
        Lv_out = Lv + Lv_out * lv_gate
        Lc_out = Lc + Lc_out  # Simpler residual connection for Lc

        return Lv_out, Lc_out, Ld_out
