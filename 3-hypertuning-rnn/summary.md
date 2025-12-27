# Summary week 3
For this exercise I followed the same pattern as for previous exercises. Starting with running a baseline.

## Baseline - GRU
The notebook was setup with a standard GRU model with one hidden layer of 64 units. This baseline started with 3 epochs. Therefore, as the notebook suggested, I increased the number of epochs to 100 for training

## GRU - High Epochs
With the rest of the settings left unchanged, the model stopped at Epoch 37 because the model didn't find any improvement for 5 epochs in a row. Meanwhile, this architecture was already able to achieve 97% on the Accuracy metric. With the base architecture already achieving such great results, I wanted to understand how a basic RNN would perform.

## RNN - High Epochs
Where the GRU rose fast in Accuracy in the few epochs, it quickly become clear that this was not the case for a vanilla RNN model as it couldn't get above 17% Accuracy. Clearly, the RNN model fails compared to the a GRU or LSTM, due to the lack of gates regulating what information should be perserved and what can be forgotton. And because this data depends on knowing what happened prior for a relatively longer period, the RNN will in no setup achieve the desired outcome.

As we learned about the paper 'Attention is all you need', I wanted to perform one more experiment as I saw this model avaialable in the `rnn_models` package.

## AttentionGRU
A GRU architecture combined with a Transformers attention mechanism. I was wondering how that would aid the GRU model. However, as the basic GRU already reached 97% I didn't think it would matter that much. This was also confirmed when the AttentionGRU only added 1% in Accurcy. Even though the model was able to converge faster (10 epcohs), this comes at the cost of extra complexity which is probably not worth the extra 1%.

# Closing remarks
Since the baseline already used a GRU architecture, it was not necessary to test a LSTM as it takes longer to compute and only has some more benefit for longer sequences. Also the width and depth remained untouched as I didn't see the need to do so with already achieving 97%, but that could be shortsighted from my end.

Find the [notebook](./notebook.ipynb) and the [instructions](./instructions.md)

[Go back to Homepage](../README.md)