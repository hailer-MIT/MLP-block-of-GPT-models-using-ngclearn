from jax import numpy as jnp, random, jit
from ngclearn import Context
from ngclearn.utils.jaxProcess import JaxProcess
from ngclearn.components import RateCell, HebbianSynapse, GaussianErrorCell, StaticSynapse
from ngclearn.utils.model_utils import layer_normalize
import ngclearn.utils.weight_distribution as dist


class SimplePCNMLP:
    """2-Layer Predictive-Coding MLP Block + Output Layer for Transformer Decoder.

    This module contains:
    1. MLP Block: Processes attention block output through 2 hidden layers
    2. Output Layer: Language modeling head that produces vocabulary logits
    
    Structure:
    - Pre-norm : Layer normalization before MLP
    - z_mlp1: first hidden layer (stateful, expansion) - receives attention output directly
      Dimension: d_ff (typically 4×d, e.g., 3072)
    - z_mlp2: second hidden layer (stateful, contraction)
      Dimension: d (back to model dimension, e.g., 768)

    - Post-norm : Layer normalization after MLP, before language head
    - z_out: output layer (stateless) - produces vocabulary logits
      Dimension: V (vocabulary size, e.g., 50257)
    
    Flow:
    attention_output → [pre-norm] → MLP (z_mlp1 → z_mlp2 → z_mlp_out) → [post-norm] → Language Head (z_out)
    
    Note: Attention output is received directly (no separate input layer needed
    since attention block is already implemented by other team members).
    
    Generative synapses (W1, W2, W_out) predict activity in the next layer.
    Gaussian error cells compute prediction errors.
    Static synapses pass feedback signals downward.
    """
    def __init__(self, dkey, input_dim=768, hidden1_dim=None, hidden2_dim=None, vocab_size=50257, 
                 learning_rate=0.001, activation="gelu", T_inference=10, tau_m=10.,
                 use_pre_norm=True, use_post_norm=False, eps=1e-5):
        """
        Args:
            dkey: JAX random key
            input_dim: Model dimension d (e.g., 768)
            hidden1_dim: Expansion dimension d_ff (e.g., 3072 = 4×768)
            vocab_size: Vocabulary size V (e.g., 50257)
            learning_rate: Learning rate for Hebbian synapses
            activation: Activation function for hidden layers (default: "gelu")
            T_inference: Number of inference steps for E-step
            tau_m: Membrane time constant for stateful layers
            use_pre_norm: Apply layer normalization before MLP (default: True)

            use_post_norm: Apply layer normalization after MLP, before language head (default: False)
            eps: Small constant for numerical stability in normalization
        """
        self.input_dim = input_dim
        # default hidden dims: d_ff = 4*d, d_ff2 = 2*d when not provided
        if hidden1_dim is None:
            hidden1_dim = 4 * input_dim
        if hidden2_dim is None:
            hidden2_dim = 2 * input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = input_dim  # MLP output dimension (d)
        self.vocab_size = vocab_size  # Vocabulary size for output layer
        self.T_inference = T_inference
        self.tau_m = tau_m
        self.use_pre_norm = use_pre_norm

        self.use_post_norm = use_post_norm
        self.eps = eps

        # Initialize learnable layer norm parameters (gamma and beta)
        # For pre-norm: normalize attention output before MLP
        # For post-norm: normalize MLP output before language head
        # Gamma (scale) initialized to 1, Beta (shift) initialized to 0
        if use_pre_norm:
            self.ln1_gamma = jnp.ones((input_dim,))  # Scale parameter
            self.ln1_beta = jnp.zeros((input_dim,))  # Shift parameter
        if use_post_norm:
            self.ln2_gamma = jnp.ones((input_dim,))  # Scale parameter
            self.ln2_beta = jnp.zeros((input_dim,))  # Shift parameter

        # split keys for components: W1, W2, W3, W_out, E2, E3, E_out
        k1, k2, k3, k4, k5, k6, k7 = random.split(dkey, 7)

        with Context("PCN_MLP_Circuit") as self.circuit:
            # Neuronal layers

            # First hidden MLP layer: stateful expansion layer (d → d_ff)
            self.z_mlp1 = RateCell("z_mlp1", n_units=self.hidden1_dim, tau_m=self.tau_m, act_fx=activation)

            # Second hidden MLP layer: stateful hidden layer (d_ff → d_ff2)
            self.z_mlp2 = RateCell("z_mlp2", n_units=self.hidden2_dim, tau_m=self.tau_m, act_fx=activation)

            # MLP output (stateless): projects back to model dimension d
            self.z_mlp3 = RateCell("z_mlp3", n_units=self.output_dim, tau_m=0., act_fx="linear")

            # Output layer: stateless layer (d dimension)
            self.z_output = RateCell("z_output", n_units=self.output_dim, tau_m=0., act_fx="linear")

            # Target layer: stateless layer that produces vocabulary logits
            self.z_target = RateCell("z_target", n_units=self.vocab_size, tau_m=0., act_fx="softmax")

            # Error cells (Gaussian errors between prediction mu and actual z)
            self.e1 = GaussianErrorCell("e1", n_units=self.hidden1_dim)
            self.e2 = GaussianErrorCell("e2", n_units=self.hidden2_dim)
            self.e3 = GaussianErrorCell("e3", n_units=self.output_dim)
            self.e_out = GaussianErrorCell("e_out", n_units=self.output_dim)
            self.e_target = GaussianErrorCell("e_target", n_units=self.vocab_size)

            # Generative synapses: 
            # W1: expands from d to d_ff (predicts z_mlp2 from z_mlp1)
            # W2: projects from d_ff to d_ff2 (predicts z_mlp3 from z_mlp2)
            # W3: projects from d_ff2 back to d (predicts z_output from z_mlp3)
            # W_out: projects from d to V (predicts z_target from z_output)
            wlb, wub = -0.1, 0.1
            self.W1 = HebbianSynapse(
                "W1", shape=(self.input_dim, self.hidden1_dim), eta=learning_rate,
                weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.),
                optim_type="adam", sign_value=-1., key=k1
            )

            self.W2 = HebbianSynapse(
                "W2", shape=(self.hidden1_dim, self.hidden2_dim), eta=learning_rate,
                weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.),
                optim_type="adam", sign_value=-1., key=k2
            )

            self.W3 = HebbianSynapse(
                "W3", shape=(self.hidden2_dim, self.output_dim), eta=learning_rate,
                weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.),
                optim_type="adam", sign_value=-1., key=k3
            )

            self.W_out = HebbianSynapse(
                "W_out", shape=(self.output_dim, self.vocab_size), eta=learning_rate,
                weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.),
                optim_type="adam", sign_value=-1., key=k4
            )

            # Feedback synapses (static): carry error signals downward (pinned to W.T)
            # shapes are transposes of forward synapse shapes
            self.E2 = StaticSynapse("E2", shape=(self.hidden2_dim, self.hidden1_dim), 
                                    weight_init=dist.uniform(amin=wlb, amax=wub), key=k4)
            self.E3 = StaticSynapse("E3", shape=(self.output_dim, self.hidden2_dim), 
                                    weight_init=dist.uniform(amin=wlb, amax=wub), key=k5)
            self.E_out = StaticSynapse("E_out", shape=(self.output_dim, self.hidden2_dim), 
                                       weight_init=dist.uniform(amin=wlb, amax=wub), key=k6)
            self.E_target = StaticSynapse("E_target", shape=(self.vocab_size, self.output_dim), 
                                          weight_init=dist.uniform(amin=wlb, amax=wub), key=k7)

            # Wiring: generative forward predictions
            # W1: attention → z_mlp1 (d → d_ff)
            # Note: W1.inputs will be set directly from attention output
            self.e1.mu << self.W1.outputs
            self.e1.target << self.z_mlp1.z

            # W2: z_mlp1 → z_mlp2 (d_ff → d_ff2)
            self.W2.inputs << self.z_mlp1.zF
            self.e2.mu << self.W2.outputs
            self.e2.target << self.z_mlp2.z

            # W3: z_mlp2 → z_mlp3 (d_ff2 → d)
            self.W3.inputs << self.z_mlp2.zF
            self.e3.mu << self.W3.outputs
            self.e3.target << self.z_mlp3.z

            # W3: z_mlp2 → z_mlp3 (d_ff → d_ff2)
            self.W3.inputs << self.z_mlp2.zF
            self.e3.mu << self.W3.outputs
            self.e3.target << self.z_mlp3.z

            # W_out: z_mlp3 → z_output (d_ff2 → d)
            self.W_out.inputs << self.z_mlp3.zF
            self.e_out.mu << self.W_out.outputs
            self.e_out.target << self.z_output.z

            # W_target: z_output → z_target (d → V)
            self.W_target = HebbianSynapse(
                "W_target", shape=(self.output_dim, self.vocab_size), eta=learning_rate,
                weight_init=dist.uniform(amin=wlb, amax=wub), bias_init=dist.constant(value=0.),
                optim_type="adam", sign_value=-1., key=k7
            )
            self.W_target.inputs << self.z_output.zF
            self.e_target.mu << self.W_target.outputs
            self.e_target.target << self.z_target.z

            # Feedback/error propagation into neuronal currents
            # z_mlp1 receives feedback from e2 (via E2) and its own local error e1
            self.E2.inputs << self.e2.dmu
            self.z_mlp1.j << self.E2.outputs
            self.z_mlp1.j_td << self.e1.dtarget

            # z_mlp2 receives feedback from e3 (via E3) and its own local error e2
            self.E3.inputs << self.e3.dmu
            self.z_mlp2.j << self.E3.outputs
            self.z_mlp2.j_td << self.e2.dtarget

            # z_mlp3 receives feedback from e_out (via E_out) and its own local error e3
            self.E_out.inputs << self.e_out.dmu
            self.z_mlp3.j << self.E_out.outputs
            self.z_mlp3.j_td << self.e3.dtarget

            # z_output receives feedback from e_target (via E_target) and its own local error e_out
            self.E_target.inputs << self.e_target.dmu
            self.z_output.j << self.E_target.outputs
            self.z_output.j_td << self.e_out.dtarget

            # z_target only receives its own local error e_target (no feedback from below)
            self.z_target.j_td << self.e_target.dtarget

            # Hebbian learning signals for synapses
            # W1.pre will be set directly from attention output
            self.W1.post << self.e1.dmu
            
            self.W2.pre << self.z_mlp1.zF
            self.W2.post << self.e2.dmu

            self.W3.pre << self.z_mlp2.zF
            self.W3.post << self.e3.dmu

            self.W_out.pre << self.z_mlp3.zF
            self.W_out.post << self.e_out.dmu

            self.W_target.pre << self.z_output.zF
            self.W_target.post << self.e_target.dmu

            # Processes: reset, advance (E-step), and learn (M-step)
            reset_process = (
                JaxProcess(name="reset_process")

                >> self.z_mlp1.reset
                >> self.z_mlp2.reset
                >> self.z_mlp3.reset
                >> self.z_output.reset
                >> self.z_target.reset
                >> self.e1.reset
                >> self.e2.reset
                >> self.e3.reset
                >> self.e_out.reset
                >> self.e_target.reset
                >> self.W1.reset
                >> self.W2.reset
                >> self.W3.reset
                >> self.W_out.reset
                >> self.W_target.reset
                >> self.E2.reset
                >> self.E3.reset
                >> self.E_out.reset
                >> self.E_target.reset
            )

            advance_pcn_state_process = (
                JaxProcess(name="advance_pcn_state_process")
                >> self.E2.advance_state
                >> self.E3.advance_state
                >> self.E_out.advance_state
                >> self.z_mlp1.advance_state
                >> self.z_mlp2.advance_state
                >> self.z_mlp3.advance_state
                >> self.z_output.advance_state
                >> self.z_target.advance_state
                >> self.W1.advance_state
                >> self.W2.advance_state
                >> self.W3.advance_state
                >> self.W_out.advance_state
                >> self.W_target.advance_state
                >> self.e1.advance_state
                >> self.e2.advance_state
                >> self.e3.advance_state
                >> self.e_out.advance_state
                >> self.e_target.advance_state
            )

            learn_process = (
                JaxProcess(name="learn_process")
                >> self.W1.evolve
                >> self.W2.evolve
                >> self.W3.evolve
                >> self.W_out.evolve
                >> self.W_target.evolve
            )

            # Expose commands on the context
            self.circuit.wrap_and_add_command(jit(reset_process.pure), name="reset")
            self.circuit.wrap_and_add_command(jit(advance_pcn_state_process.pure), name="advance_pcn")
            self.circuit.wrap_and_add_command(jit(learn_process.pure), name="learn")

            # Dynamic commands: clamp attention input and clamp target output
            @Context.dynamicCommand
            def set_attention_error(x):
                """Set attention output as top-down error to z_mlp1."""
                self.z_mlp1.j_td.set(x)

            @Context.dynamicCommand
            def clamp_target_output(y):
                """Clamp target vocabulary (one-hot or logits) to output layer during training."""
                self.z_target.j.set(y)

    def _layer_norm(self, x, gamma, beta):
        """Apply layer normalization: norm(x) = gamma * (x - μ) / (σ + ε) + beta
        
        Normalizes over the last dimension (feature dimension).
        Works for both 2D [B, d] and 3D [B, T, d] tensors.
        """
        # Compute mean and std over last dimension (feature dimension)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.eps)
        
        # Normalize: (x - mean) / std
        normalized = (x - mean) / std
        
        # Scale and shift: gamma * normalized + beta
        # gamma and beta broadcast to match x shape
        return gamma * normalized + beta

    def forward(self, attention_output):
        """Forward pass: process attention output through MLP and output layer.
        
        This is the main method for inference in the transformer architecture.
        Processes attention block output through MLP and produces vocabulary logits.
        
        Flow:
        1. Pre-norm : normalize attention output
        2. MLP Block: z_mlp1 → z_mlp2 → z_mlp_out
        3. Post-norm : normalize before language head
        4. Language Head: z_mlp_out → z_out

        Args:
            attention_output: Input tensor from attention block
               Shape: (batch, seq_len, input_dim)

        Returns:
            Vocabulary logits from output layer.
            Shape: (batch, seq_len, vocab_size) or (batch, vocab_size)
        """
        # Step 1: Pre-norm (normalize attention output before MLP)
        if self.use_pre_norm:
            attention_output = self._layer_norm(attention_output, self.ln1_gamma, self.ln1_beta)
        
        # Step 2: Process through MLP block
        self.circuit.reset()
        self.circuit.set_attention_error(attention_output)

        # Pin feedback weights to current W transposes
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))
        self.E_out.weights.set(jnp.transpose(self.W_out.weights.value))

        # Initialize stateful hidden layers to zeros and run E-step settling
        self.z_mlp1.z.set(jnp.zeros_like(self.z_mlp1.z.value))
        self.z_mlp2.z.set(jnp.zeros_like(self.z_mlp2.z.value))
        self.z_mlp_out.z.set(jnp.zeros_like(self.z_mlp_out.z.value))
        for ts in range(self.T_inference):
            self.circuit.advance_pcn(t=ts, dt=1.)

        # Step 3: Get MLP output
        mlp_output = self.z_mlp_out.zF.value
        
        # Step 4: Post-norm (normalize MLP output before language head)
        if self.use_post_norm:
            mlp_output = self._layer_norm(mlp_output, self.ln2_gamma, self.ln2_beta)
        
        # Step 5: Process through language head
        self.W_out.inputs.set(mlp_output)
        self.W_out.pre.set(mlp_output)
        self.W_out.advance_state(t=0., dt=1.)
        self.e_out.advance_state(t=0., dt=1.)
        self.z_out.advance_state(t=0., dt=1.)

        # Return vocabulary logits from output layer
        return self.z_out.zF.value

    def predict(self, attention_output):
        """Alias for forward() for backward compatibility."""
        return self.forward(attention_output)

    def train_step(self, attention_output, target_logits=None):
        """One PCN training step: inference (E-step) then learning (M-step).
        
        Args:
            attention_output: Input tensor from attention block
               Shape: (batch, seq_len, input_dim) or (batch, input_dim)
            target_logits: Optional target vocabulary logits for training
               Shape: (batch, seq_len, vocab_size) or (batch, vocab_size)
               If None, only MLP layers are trained (unsupervised)

        Returns:
            output_logits: Vocabulary logits from z_out
            EFE: Expected Free Energy (sum of local losses)
        """
        # Step 1: Pre-norm (normalize attention output before MLP)
        if self.use_pre_norm:
            attention_output = self._layer_norm(attention_output, self.ln1_gamma, self.ln1_beta)
        
        # Step 2: Process through MLP block
        self.circuit.reset()
        self.circuit.set_attention_error(attention_output)
        
        if target_logits is not None:
            self.circuit.clamp_target_output(target_logits)

        # Pin feedback weights to current W transposes
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))
        self.E_out.weights.set(jnp.transpose(self.W_out.weights.value))

        # Initialize stateful hidden layers to zeros and run E-step
        self.z_mlp1.z.set(jnp.zeros_like(self.z_mlp1.z.value))
        self.z_mlp2.z.set(jnp.zeros_like(self.z_mlp2.z.value))
        self.z_mlp_out.z.set(jnp.zeros_like(self.z_mlp_out.z.value))
        for ts in range(self.T_inference):
            self.circuit.advance_pcn(t=ts, dt=1.)

        # Step 3: Get MLP output
        mlp_output = self.z_mlp_out.zF.value
        
        # Step 4: Post-norm (normalize MLP output before language head)
        if self.use_post_norm:
            mlp_output = self._layer_norm(mlp_output, self.ln2_gamma, self.ln2_beta)
        
        # Step 5: Update language head
        self.W_out.inputs.set(mlp_output)
        self.W_out.pre.set(mlp_output)
        self.W_out.advance_state(t=0., dt=1.)
        self.e_out.advance_state(t=0., dt=1.)
        self.z_out.advance_state(t=0., dt=1.)

        # Expected Free Energy (sum of local losses)
        EFE = self.e1.L.value + self.e2.L.value + self.e3.L.value + self.e_out.L.value

        # M-step: update synapses
        self.circuit.learn(t=0., dt=1.)

        return self.z_out.zF.value, EFE


# Example usage
def main():
    """Example initialization of MLP block + Output layer.
    
    Note: Dataset loading, training, and prediction are handled by other team members.
    This is just a simple example to demonstrate model initialization.
    """
    key = random.PRNGKey(0)

    # Model configuration (example values)
    model_dim = 768  # Model dimension d (e.g., GPT-2 Small)
    d_ff = 3072  # Expansion dimension (4× model_dim)
    vocab_size = 50257  # Vocabulary size (e.g., GPT-2)
    
    # Initialize MLP block + Output layer
    pcn_mlp = SimplePCNMLP(
        key,
        input_dim=model_dim,  # Model dimension d
        hidden1_dim=d_ff,     # Expansion dimension d_ff
        vocab_size=vocab_size,  # Vocabulary size V
        T_inference=20,
        tau_m=20.
    )

    print("✓ MLP block + Output layer initialized successfully!")
    print(f"  Model dimension: {model_dim}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Vocabulary size: {vocab_size}")
    print("\nReady for integration into transformer architecture!")
    print("  - Use pcn_mlp.forward(attention_output) for inference")
    print("  - Use pcn_mlp.train_step(attention_output, target_logits) for training")

if __name__ == "__main__":
    main()
