A demo for word-level generative adversarial model.

The dimension of z is the same as the embedding dimension, and we directly feed z to the first LSTM cell of the generator and do the sentence generation.

However, the components of the generator and the discriminator have various versions:
* textGAN: CNN as discriminator, LSTM as generator and use a standard objective function.
* textGANV1: LSTM as discriminator, LSTM as generator and use a standard objective function.
* textGANV2: Similar to textGANV1, but we use WGAN.
