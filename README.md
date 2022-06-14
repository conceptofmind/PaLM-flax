<img src="./palm.gif" width="450px"></img>

## Acknowledgements:
- [Dr. Phil Wang](https://github.com/lucidrains/)

## PaLM-flax
Implementation of the Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a> - in Jax using [Flax](https://github.com/google/flax).

## Usage

The model does not require `vmap` to train.

```python
import jax
import numpy as np

key = jax.random.PRNGKey(0)

seq = jax.random.randint(key, (1, 1024), 0, 20000)

model = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2)}

params = model.init(init_rngs, seq)
output = model.apply(params, seq)
print(output.shape) # (1, 1024, 20000)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}") # 55073280
```

## Author:
- Enrico Shippole

## Citations:
```bibtex
@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.5.0},
  year = {2020},
}
```
