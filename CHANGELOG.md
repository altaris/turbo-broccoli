Changelog
=========

# `v4.0.0`

- `safetensors` is not required for (de)serializing `numpy.ndarray`,
  `torch.Tensor`, and `tensorflow.Tensor`.
- Before `v4`, JSON document looked like
  ```json
  {
    "__TYPE__": {
      ...
    }
  }
  ```
  and
  ```json
  {
    "__TYPE__": {
      "__type__": "SUBTYPE",
    }
  }
  ```
  now they look like
  ```json
  {
    "__type__": "TYPE",
    ...
  }
  ```
  and
  ```json
  {
    "__type__": "TYPE.SUBTYPE",
    ...
  }
  ```
  This declutters the overall structure of the JSON documents.

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
