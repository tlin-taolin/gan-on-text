Clarify the model:

* textGANV0 -- seq2seq chatbot.
    * classical seq2seq model.
    * input: enc, dec, target.
    * loss: sparse_softmax_cross_entropy_with_logits
* textGANV1 -- improved WGAN for sentence generation. no noise and should fail.
    * RNN as the generator, CNN as the discriminator.
    * directly use previous output to approximate the embedding, and then use it as the input.
    * input: first element of the dec, target.
    * improved WGAN.
* textGANV2 -- improved WGAN for sen2seq chatbot.
    * seq2seq as the generator, CNN as the discriminator.
    * input: enc, dec, target.
    * improved WGAN.
* textGANV3 -- improved WGAN for sentence generation.
    * the main purpose of this model is to let the `free running` mode indistinguishable from the `teacher forcing` mode.

---

Justify the dataset:

* dataLoaderBBT -- use it for textGANV0, textGANV2.
* dataLoaderBBTV1 -- use it for textGANV1.
