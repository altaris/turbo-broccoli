Changelog
=========

# `v2.0.0`

* `turbo-broccoli` now uses [Huggingface's
  `safetensors`](https://huggingface.co/docs/safetensors/index) to store
  `np.ndarray` and `tf.Tensor`.
* Added support for `torch.Tensor`, also serialized using `safetensors`.
