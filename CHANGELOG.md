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
- Introduction of (de)serialization context objects
  (`turbo_broccoli.context.Context`), which contain various information and
  parameters about the ongoing (de)serialization operation. You can use it when
  calling `turbo_broccoli.from_json` or `turbo_broccoli.to_json`. For
  convenience, `turbo_broccoli.save_json` and `turbo_broccoli.load_json` take
  the context parameter's as kwargs.
- Removal of `turbo_broccoli.environment`. Use `turbo_broccoli.context.Context`
  instead.
- Sentinel booleans like `HAS_NUMPY`, `HAS_PANDA` etc. are now in
  `turbo_broccoli.custom`.
- Deleted `turbo_broccoli.guard.guarded_call` and
  `turbo_broccoli.guard.produces_document`. The iterable version of
  `turbo_broccoli.guard.GuardedBlockHandler.guard` no longer exist.
- Removal of the CLI.

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
