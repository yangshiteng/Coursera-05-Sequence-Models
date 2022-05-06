# Week 1 - Recurrent Neural Networks (RNN)

Welcome to this fifth course on deep learning. In this course, you learn about sequence models, one of the most exciting areas in deep learning. Models like recurrent neural networks or RNNs have transformed speech recognition, natural language processing and other areas. And in this course, you learn how to build these models for yourself.

![image](https://user-images.githubusercontent.com/60442877/164052425-ec98138a-ac97-4928-a2cc-650b4906b1db.png)

## Notation

![image](https://user-images.githubusercontent.com/60442877/164153556-98f27d00-1a2f-4ffa-b983-f5773818f580.png)

![image](https://user-images.githubusercontent.com/60442877/164153588-f744c2b3-0450-435f-bdde-ccf9e05ee154.png)

## Why not a standard Network?

![image](https://user-images.githubusercontent.com/60442877/164154296-5d7379d8-904c-4890-a3c8-707f8c7d6d16.png)

## Recurrent Neural Networks

![image](https://user-images.githubusercontent.com/60442877/164875180-bd0eb4d9-25e1-423a-9629-81a94e7ed201.png)

So, what is a recurrent neural network? Let's build one up. So if you are reading the sentence from left to right, the first word you will read is the some first words say X1, and what we're going to do is take the first word and feed it into a neural network layer. I'm going to draw it like this. So there's a hidden layer of the first neural network and we can have the neural network maybe try to predict the output. So is this part of the person's name or not. And what a recurrent neural network does is, when it then goes on to read the second word in the sentence, say x2, instead of just predicting y2 using only X2, it also gets to input some information from whether the computer that time step one. So in particular, deactivation value from time step one is passed on to time step two. Then at the next time step, recurrent neural network inputs the third word X3 and it tries to output some prediction, Y hat three and so on up until the last time step where it inputs x_TX and then it outputs y_hat_ty. At least in this example, Ts is equal to ty and the architecture will change a bit if tx and ty are not identical. So at each time step, the recurrent neural network that passes on as activation to the next time step for it to use. And to kick off the whole thing, we'll also have some either made-up activation at time zero, this is usually the vector of zeros. Some researchers will initialized a_zero randomly. You have other ways to initialize a_zero but really having a vector of zeros as the fake times zero activation is the most common choice.

![image](https://user-images.githubusercontent.com/60442877/164878303-1e7bd849-1522-445b-9443-63139fec28ed.png)

Now, one weakness of this RNN is that it only uses the information that is earlier in the sequence to make a prediction. In particular, when predicting y3, it doesn't use information about the words X4, X5, X6 and so on. So this is a problem because if you are given a sentence, "He said Teddy Roosevelt was a great president." In order to decide whether or not the word Teddy is part of a person's name, it would be really useful to know not just information from the first two words but to know information from the later words in the sentence as well because the sentence could also have been, "He said teddy bears they're on sale." So given just the first three words is not possible to know for sure whether the word Teddy is part of a person's name. In the first example, it is. In the second example, it is not. But you can't tell the difference if you look only at the first three words. So one limitation of this particular neural network structure is that the prediction at a certain time uses inputs or uses information from the inputs earlier in the sequence but not information later in the sequence. We will address this in a later video where we talk about bi-directional recurrent neural networks or BRNNs.

![image](https://user-images.githubusercontent.com/60442877/164878532-d271bf56-41d6-4d74-ae9d-8405966d21c9.png)

## Backpropagation Through Time

![image](https://user-images.githubusercontent.com/60442877/165147316-9e841653-ca10-4e9a-a507-86f6bce3a91e.png)

## Different Types of RNNs

- Many to Many RNN architecture (input and output same length)
- Many to Many RNN architecture (input and output different length) (machine translation)
- Many to One RNN architecture
- One to Many RNN architecture
- One to one RNN architecture

![image](https://user-images.githubusercontent.com/60442877/165148531-62b03747-7f67-4a4e-a47b-3fb81cc55c09.png)
![image](https://user-images.githubusercontent.com/60442877/165149191-dfa94e5c-3cd8-4529-8b90-e89a145b1f98.png)
![image](https://user-images.githubusercontent.com/60442877/165149695-092254e6-f929-4cfa-acb0-2ed460858c79.png)

## Language Modeling

![image](https://user-images.githubusercontent.com/60442877/166125156-a0115dbc-a8f1-485b-afad-233a25e33aac.png)

![image](https://user-images.githubusercontent.com/60442877/166125517-cd1ad22d-a878-4718-83ab-ca15b3c6474b.png)

## Sampling Novel Sequence

After you train a sequence model, one of the ways you can informally get a sense of what is learned is to have a sample novel sequences. So remember that a sequence model, models the chance of any particular sequence of words as follows, and so what we like to do is sample from this distribution to generate noble sequences of words.

![image](https://user-images.githubusercontent.com/60442877/166341387-e2177753-3d1f-44a5-9dea-ab2b53fc9521.png)

So the network was trained using this structure shown at the top. But to sample, you do something slightly different, so what you want to do is first sample what is the first word you want your model to generate. And so for that you input the usual x1 equals 0, a0 equals 0. And now your first time stamp will have some max probability over possible outputs. So what you do is you then randomly sample according to this soft max distribution. So what the soft max distribution gives you is it tells you what is the chance that it refers to this a, what is the chance that it refers to this Aaron? What's the chance it refers to Zulu, what is the chance that the first word is the Unknown word token. Maybe it was a chance it was a end of sentence token. And then you take this vector and use, for example, the numpy command np.random.choice to sample according to distribution defined by this vector probabilities, and that lets you sample the first words. Next you then go on to the second time step, and now remember that the second time step is expecting this y1 as input. But what you do is you then take the y1 hat that you just sampled and pass that in here as the input to the next timestep. So whatever works, you just chose the first time step passes this input in the second position, and then this soft max will make a prediction for what is y hat 2.

## Character-level language model

Now, so far we've been building a words level RNN, by which I mean the vocabulary are words from English. Depending on your application, one thing you can do is also build a character level RNN. So in this case your vocabulary will just be the alphabets. Up to z, and as well as maybe space, punctuation if you wish, the digits 0 to 9. And if you want to distinguish the uppercase and lowercase, you can include the uppercase alphabets as well, and one thing you can do as you just look at your training set and look at the characters that appears there and use that to define the vocabulary. And if you build a character level language model rather than a word level language model, then your sequence y1, y2, y3, would be the individual characters in your training data, rather than the individual words in your training data. So for our previous example, the sentence cats average 15 hours of sleep a day. In this example, c would be y1, a would be y2, t will be y3, the space will be y4 and so on.

![image](https://user-images.githubusercontent.com/60442877/166341747-db8e284e-be63-41e7-b151-d76fa47873ed.png)

Using a character level language model has some pros and cons. One is that you don't ever have to worry about unknown word tokens. In particular, a character level language model is able to assign a sequence like mau, a non-zero probability. Whereas if mau was not in your vocabulary for the word level language model, you just have to assign it the unknown word token. But the main disadvantage of the character level language model is that you end up with much longer sequences. So many english sentences will have 10 to 20 words but may have many, many dozens of characters. And so character language models are not as good as word level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence. And character level models are also just more computationally expensive to train. So the trend I've been seeing in natural language processing is that for the most part, word level language model are still used, but as computers gets faster there are more and more applications where people are, at least in some special cases, starting to look at more character level models. But they tend to be much hardware, much more computationally expensive to train, so they are not in widespread use today. Except for maybe specialized applications where you might need to deal with unknown words or other vocabulary words a lot. Or they are also used in more specialized applications where you have a more specialized vocabulary. So under these methods, what you can now do is build an RNN to look at the purpose of English text, build a word level, build a character language model, sample from the language model that you've trained.

## Vanishing Gradients with RNNs

You've learned about how RNNs work and how they can be applied to problems like name entity recognition as well as to language modeling. You saw how back propagation can be used to train an RNN. It turns out that one of the problems of the basic RNN algorithm is that it runs into vanishing gradient problems.

![image](https://user-images.githubusercontent.com/60442877/166344235-4f49ff9d-bf3f-496d-ad31-f1634a4d40fd.png)

Let's discuss that and in the next few videos we'll talk about some solutions that will help to address this problem. You've seen pictures of RNNs that look like this. Let's take a language modeling example. Let's say you see this sentence. The cat, which already ate and maybe already ate a bunch of food that was delicious, dot dot, dot, dot, dot was full. To be consistent is because the cat is singular, it should be the cat was where there was the cats, which already ate a bunch of food was delicious and the apples and pears and so on were full. To be consistent, it should be cat was or cats were. This is one example of when language can have very long-term dependencies where it worded as much earlier can affect what needs to come much later in the sentence. But it turns out that the basic RNN we've seen so far is not very good at capturing very long-term dependencies. To explain why, you might remember from our earlier discussions of training very deep neural networks that we talked about the vanishing gradients problem. This is a very, very deep neural network, say 100 years or even much deeper. Then you would carry out forward prop from left to right and then backprop. We said that if this is a very deep neural network, then the gradient from this output y would have a very hard time propagating back to affect the weights of these earlier layers, to affect the computations of the earlier layers. For an RNN with a similar problem, you have forward prop going from left to right and then backprop going from right to left. It can be quite difficult because of the same vanishing gradients problem for the outputs of the errors associated with the later timesteps to affect the computations that are earlier. In practice, what this means is it might be difficult to get a neural network to realize that it needs to memorize. Did you see a singular noun or a plural noun so that later on in the sequence it can generate either was or were, depending on whether it was singular or plural. Notice that in English this stuff in the middle could be arbitrarily long. You might need to memorize the singular plural for a very long time before you get to use that bit of information. Because of this problem, the basic RNN model has many local influences, meaning that the output y hat three is mainly influenced by values close to y hat three and a value here is mainly influenced by inputs that are somewhat close. It's difficult for the output here to be strongly influenced by an input that was very early in the sequence. This is because whatever the output is, whether this got it right, this got it wrong, it's just very difficult for the error to backpropagate all the way to the beginning of the sequence, and therefore to modify how the neural network is doing computations earlier in the sequence. This is a weakness of the basic RNN algorithm, one which will to address in the next few videos. But if we don't address it, then RNNs tend not to be very good at capturing long-range dependencies. 

## Gated Recurrent Unit (GRU) (capture very long-term dependency)

You've seen how a basic RNN works. In this video, you learn about the gated recurrent unit, which has a modification to the RNN hidden layer that makes it much better at capturing long-range connections and helps a lot with the vanishing gradient problems.

![image](https://user-images.githubusercontent.com/60442877/166345086-9b319191-6d31-4213-b396-3e858237566b.png)

![image](https://user-images.githubusercontent.com/60442877/166390063-40eb7c15-7a27-4f57-9b5e-92473f9fe816.png)

![image](https://user-images.githubusercontent.com/60442877/166390895-d711cca9-d77b-4f74-984a-538609e7dd29.png)

- Gamma u is the update gate
- Gamma r is the relevant gate

## Long Short Term Memory (LSTM) (capture very long-term dependency)

In the last video, you learn about the GRU, the Gated Recurring Unit and how that can allow you to learn very long range connections in a sequence. The other type of unit that allows you to do this very well is the LSTM or the long short term memory units. And this is even more powerful than the GRU, let's take a look.

The LSTM is an even slightly more powerful and more general version of the GRU. And notice that for the LSTM, we will no longer have the case that a_t is equal to c_t.

- Gamma_f is called "Forget Gate"
- Gamma_o is called "Output Gate"

![image](https://user-images.githubusercontent.com/60442877/166401763-38d771fc-7b9c-427e-acb9-3ef7820de466.png)

![image](https://user-images.githubusercontent.com/60442877/166401611-c908b92c-5e74-41ea-9b16-95ff29063b62.png)

So that's it for the LSTM, when should you use a GRU and when should you use an LSTM. There is a widespread consensus in this. And even though I presented GRUs first in the history of deep learning, LSTMs actually came much earlier and then GRUs were relatively recent invention that were maybe derived as partly a simplification of the more complicated LSTM model. Researchers have tried both of these models on many different problems and on different problems the different algorithms will win out. So there isn't a universally superior algorithm, which is why I want to show you both of them. But I feel like when I am using these, the advantage of the GRU is that it's a simpler model. And so it's actually easier to build a much bigger network only has two gates, so computation runs a bit faster so it scales the building, somewhat bigger models. But the LSTM is more powerful and more flexible since there's three gates instead of two. If you want to pick one to use, I think LSTM has been the historically more proven choice. So if you had to pick one, I think most people today will still use the LSTM as the default first thing to try. Although I think the last few years GRUs have been gaining a lot of momentum and I feel like more and more teams are also using GRUs because they're a bit simpler but often were, just as well and it might be easier to scale them to even bigger problems. So that's it for LSTMs with either GRUs or LSTMS, you'll be able to build new networks that can capture much longer range dependencies.

## Bidirectional RNN

By now you've seen most of the key building blocks of our RNN. But there are just two more ideas that let you build much more powerful models. One is bidirectional RNN, which lets you at the point in time to take information from both earlier and later in the sequence. So talk about that in this video. And the second is deep RNN, which you see in the next video.

![image](https://user-images.githubusercontent.com/60442877/167201765-6e8dc82e-ac43-4492-b4e5-a63a1fdd4447.png)

So to motivate bidirectional RNN, let's look at this network which you've seen a few times before in the context of named entity recognition. And one of the problems of this network is that, to figure out whether the third word Teddy is a part of a person's name, it's not enough to just look at the first part of the sentence. So to tell if y3 should be 01, you need more information than just the first few words. Because the first three words doesn't tell you if they're talking about Teddy bears, or talk about the former US President, Teddy Roosevelt. So this is a unidirectional only RNN

![image](https://user-images.githubusercontent.com/60442877/167202973-4565845d-5459-496d-b17c-f64f6d0e6af5.png)

So this is the bidirectional recurrent neural network. And these blocks here can be not just the standard RNN block, but they can also be GRU blocks, or LSTM blocks. In fact, for a lot of NLP problems, for a lot of text or natural language processing problems, a bidirectional RNN with a LSTM appears to be commonly used. So, if you have an NLP problem, and you have a complete sentence, you're trying to label things in the sentence, a bidirectional RNN with LSTM blocks both forward and backward would be a pretty reasonable first thing to try. So that's it for the bidirectional RNN. And this is a modification they can make to the basic RNN architecture, or the GRU, or the LSTM. And by making this change, you can have a model that uses RNN, or GRU, LSTM, and is able to make predictions anywhere even in the middle of the sequence, but take into account information potentially from the entire sequence. The disadvantage of the bidirectional RNN is that, you do need the entire sequence of data before you can make predictions anywhere. So, for example, if you're building a speech recognition system then BRNN will let you take into account the entire speech utterance. But if you use this straightforward implementation, you need to wait for the person to stop talking to get the entire utterance before you can actually process it, and make a speech recognition prediction. So for the real time speech recognition applications, there is somewhat more complex models as well rather than just using the standard bi-directional RNN as you're seeing here. But for a lot of natural language processing applications where you can get the entire sentence all at the same time, the standard BRNN algorithm is actually very effective. 

## Deep RNN

The different versions of RNNs you've seen so far will already work quite well by themselves. But for learning very complex functions sometimes is useful to stack multiple layers of RNNs together to build even deeper versions of these models.

![image](https://user-images.githubusercontent.com/60442877/167207689-18dd38a2-5178-41af-a54b-3f3b4fcbd581.png)

So this will be a Neural Network with three hidden layers. But then you might take the output here, let's get rid of this, and then just have a bunch of deep layers that are not connected horizontally but have a deep network here that then finally predicts y<1>. And you can have the same deep network here that predicts y<2>. So this is a type of network architecture that we're seeing a little bit more where you have three recurrent units that connected in time, followed by a network, followed by a network after that, as we seen for y<3> and y<4>, of course. There's a deep network, but that does not have the horizontal connections. So that's one type of architecture we seem to be seeing more of. And quite often, these blocks don't just have to be standard RNN, the simple RNN model. They can also be GRU blocks LSTM blocks. And finally, you can also build deep versions of the bidirectional RNN. Because deep RNNs are quite computationally expensive to train, there's often a large temporal extent already, though you just don't see as many deep recurrent layers. This has, I guess, three deep recurrent layers that are connected in time. You don't see as many deep recurrent layers as you would see in a number of layers in a deep conventional neural network.


