**Common Types of Transformers**

1. Transformer Encoder (eg. BERT)
   1. Uses the encoder side of the Transformer architecture which uses  Unmasked Self Attention.
   2. Useful if we are interested in the final layer token representation to solve downstream NLP tasks.
   3. Not useful for Text Generation task.
2. Transformer Decoder (eg. Transformer LM,GPT-2)
   1. Uses Masked Self Attention.
   2. Uses the decoder side of the Transformer architecture.
   3. Useful for generating text trained via Language Modeling objective
3. Transformer Encoder/Decoder (eg. T5)
   1. Uses both the encoder and decoder side of the Architecture.
   2. Useful for conditional text generation.
4. Prefix LM
   1. It is an alternative to Encoder+Decoder approach with just 1 model instead of 2.
   2. Source and the target sentences are concatenated together with a <SOS> token and fed into the model.
   3. Use of Partially masked self attention as the masking would be applicable to only the words after current attending word of the target sentence in the model. Source sentence will always be unmasked.
   4. Predictions done only on the target sequence to generate next token predictions.





**ELMO - Embeddings from Language Model**(2018)

1. Pretrain an RNN LM on lots of data,1 Billions words.

2. Freeze LM parameters,use its representation(hidden state) as inputs to a task specific model.

3. Used two unidirectional LM , Forward LM , Backward LM combined via concatenation.

   

**ELMO To Bert (2019)**

1. 2 unidirectional LMs -> 1 masked LM
2. Recurrent NNs to Transformers
3. Freezing the LM to finetuning the LM.
4. Pretrain LM on a way more data ,way bigger model.



**Masked LM**

1. Input is a sequence where some tokens have been randomly masked.
2. Goal is to predict identify of the masked tokens.
3. Use of Unmasked Transformer Block.



Bert

1. Masked LM model with mask % is 15%.
2. Pretraining task : Masked LM and next sentence prediction.
3. Use it for different fine-tuning task eg. Text Classification,Question Answering
4. For text classification, 
   1. [CLS] is a special token used for fine tuning the BERT's parameters.
   2. Softmax Layer trained from scratch to predict the class label which takes the CLS token and project it into n dimensions,(representing n classes for text classification).
5. For sentence pair classification
   1. Given two sentences(s1,s2), model must figure out if s2 {entails,contradicts,neutral}to s1.
   2. Both the sentences are concatenated via [SEP] token as input to the model.
   3. Softmax Layer trained from scratch to predict the class label which takes the CLS token and project it into n dimensions,(n=3 classes for this case).
   4. Backpropogation in  [SEP] token tells the model,s1 before the [SEP] has  different meaning when compared to s2 after the [SEP] token.
   5. [SEP] token can be extended for concatenating more than to sentences and they would be unique due to positional embedding.
6. For Extractive Question Answering
   1. Input - Question and the passage
   2. Goal - Predict a contiguous span of text from the passage that answers the question.
   3. Two binary classifiers: 
      1. Predict whether the token is the start of the answer.
      2. Predict whether it is the end of the answer.
   4. Answer span is selected by finding the span that maximizes P(start(i)).P(end(j))
      1. excludes span where j < i.
      2. excludes span longer than some threshold.
      3.  

**Advanced Variants of Bert**

1. Pretraining improvements by adding more data - ROBERTA 
2. Longer sequence during pretraining 
   1. Bert - 512 token
   2. XLNet - 900 token
3. More Pretraining Objective
   1. ELECTRA
4. Smaller models
   1. Albert,DistilBert,TinyBert



**ROBERTA**

Modifications as compared to BERT.

1. Train with bigger batches.
2. Gradient Accumulation over batches,to bypass GPU mem. limitations
3. Has no pretraining task for CLS.
4. Pretrained on more data.
   1. 16Gb to 160 Gb
5. Pretrained for Longer.





