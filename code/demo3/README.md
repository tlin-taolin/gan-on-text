A demo for word-level generative adversarial model.

The following option uses soft-softmax to generate continuous onehot vector. Based on how we feed the latent variable z, we have the options below:
* The dimension of z is the same as the embedding dimension, and we only feed z to the first LSTM cell of the generator and do the sentence generation.
    * textGAN: CNN as discriminator, LSTM as generator and use a standard objective function.
    * textGANV1: LSTM as discriminator, LSTM as generator and use a standard objective function.
    * textGANV2: Similar to textGANV1, but here we use WGAN.
* The dimension of z is the same as the embedding dimension, and we not only feed z to the first LSTM cell of the generator, but also to the subsequent input of the LSTM cells.
    * textGANV3: The overall architecture is similar to textGANV2.
* We feed z as the first latent state to the generator. The whole sentence is generated from a <go> symbol.
    * textGANV4
