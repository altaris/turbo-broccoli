Changelog
=========

# `v2.1.0`

- `sklearn` (de)serialization of most estimator classes, see [the list of
  supported
  types](https://altaris.github.io/turbo-broccoli/turbo_broccoli.html#supported-types)
  for more details.

- Guarded methods and guarded calls, see
  [here](https://altaris.github.io/turbo-broccoli/turbo_broccoli.html#guarded-calls)
  for some basic usage.

# `v2.0.0`

- `turbo-broccoli` now uses [Huggingface's
  `safetensors`](https://huggingface.co/docs/safetensors/index) to store
  `np.ndarray` and `tf.Tensor`.

- Added support for `torch.Tensor`, also serialized using `safetensors`.
