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

LSTM, which is better at addressing vanishing gradients. The LSTM is better able to remember a piece of information and save it for many time steps.

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

# Week 2 - Word Representation

Hello, and welcome back. Last week, we learned about RNNs, GRUs, and LSTMs. In this week, you see how many of these ideas can be applied to NLP, to Natural Language Processing, which is one of the features of AI because it's really being revolutionized by deep learning. One of the key ideas you learn about is word embeddings, which is a way of representing words to let your algorithms automatically understand analogies like that, man is to woman, as king is to queen, and many other examples. And through these ideas of word embeddings, you'll be able to build NLP applications, even with models the size of, usually of relatively small label training sets. Finally towards the end of the week, you'll see how to debias word embeddings. That's to reduce undesirable gender or ethnicity or other types of bias that learning algorithms can sometimes pick up. 

![image](https://user-images.githubusercontent.com/60442877/167533212-db6121f5-4526-4a5b-8ca9-d3636bd0a456.png)

So far, we've been representing words using a vocabulary of words, and a vocabulary from the previous week might be say, 10,000 words. And we've been representing words using a one-hot vector. So for example, if man is word number 5391 in this dictionary, then you represent him with a vector with one in position 5391.

One of the weaknesses of this representation is that it treats each word as a thing unto itself, and it doesn't allow an algorithm to easily generalize the cross words. For example, let's say you have a language model that has learned that when you see I want a glass of orange blank. Well, what do you think the next word will be? Very likely, it'll be juice. But even if the learning algorithm has learned that I want a glass of orange juice is a likely sentence, if it sees I want a glass of apple blank. As far as it knows the relationship between apple and orange is not any closer as the relationship between any of the other words man, woman, king, queen, and orange. And so, it's not easy for the learning algorithm to generalize from knowing that orange juice is a popular thing, to recognizing that apple juice might also be a popular thing or a popular phrase. And this is because the any product between any two different one-hot vector is zero. If you take any two vectors say, queen and king and any product of them, the end product is zero. If you take apple and orange and any product of them, the end product is zero. And you couldn't distance between any pair of these vectors is also the same. So it just doesn't know that somehow apple and orange are much more similar than king and orange or queen and orange.

## Featurized Representation: word embedding

![image](https://user-images.githubusercontent.com/60442877/167534327-51d9a3ac-9553-4be4-9310-1de77d139c4d.png)

Now, if you use this representation to represent the words orange and apple, then notice that the representations for orange and apple are now quite similar. Some of the features will differ because of the color of an orange, the color an apple, the taste, or some of the features would differ. But by a large, a lot of the features of apple and orange are actually the same, or take on very similar values. And so, this increases the odds of the learning algorithm that has figured out that orange juice is a thing, to also quickly figure out that apple juice is a thing. So this allows it to generalize better across different words. So over the next few videos, we'll find a way to learn words embeddings. We just need you to learn high dimensional feature vectors like these, that gives a better representation than one-hot vectors for representing different words. And the features we'll end up learning, won't have a easy to interpret interpretation like that component one is gender, component two is royal, component three is age and so on. Exactly what they're representing will be a bit harder to figure out. But nonetheless, the featurized representations we will learn, will allow an algorithm to quickly figure out that apple and orange are more similar than say, king and orange or queen and orange.

![image](https://user-images.githubusercontent.com/60442877/167534955-c5cae96b-86b9-48ff-b688-a6eccd38d7fc.png)

Word embeddings has been one of the most important ideas in NLP, in Natural Language Processing. In this video, you saw why you might want to learn or use word embeddings. In the next video, let's take a deeper look at how you'll be able to use these algorithms, to build NLP algorithms.

## Transfer Leanring and Word Embeddings

In the last video, you saw what it might mean to learn a featurized representations of different words. In this video, you see how we can take these representations and plug them into NLP applications. 

Continuing with the named entity recognition example, if you're trying to detect people's names. Given a sentence like Sally Johnson is an orange farmer, hopefully, you'll figure out that Sally Johnson is a person's name, hence, the outputs 1 like that. 

![image](https://user-images.githubusercontent.com/60442877/167545044-c217a647-e115-4c3e-b4fa-9bf86ce5f84f.png)

But if you can now use the featurized representations, the embedding vectors that we talked about in the last video. Then after having trained a model that uses word embeddings as the inputs, if you now see a new input, Robert Lin is an apple farmer. Knowing that orange and apple are very similar will make it easier for your learning algorithm to generalize to figure out that Robert Lin is also a human, is also a person's name. One of the most interesting cases will be, what if in your test set you see not Robert Lin is an apple farmer, but you see much less common words? What if you see Robert Lin is a durian cultivator?

A durian is a rare type of fruit, popular in Singapore and a few other countries. But if you have a small label training set for the named entity recognition task, you might not even have seen the word durian or seen the word cultivator in your training set. I guess technically, this should be a durian cultivator. But if you have learned a word embedding that tells you that durian is a fruit, so it's like an orange, and a cultivator, someone that cultivates is like a farmer, then you might still be generalize from having seen an orange farmer in your training set to knowing that a durian cultivator is also probably a person. So one of the reasons that word embeddings will be able to do this is the algorithms to learning word embeddings can examine very large text corpuses, maybe found off the Internet. So you can examine very large data sets, maybe a billion words, maybe even up to 100 billion words would be quite reasonable. So very large training sets of just unlabeled text.

And by examining tons of unlabeled text, which you can download more or less for free, you can figure out that orange and durian are similar. And farmer and cultivator are similar, and therefore, learn embeddings, that groups them together. Now having discovered that orange and durian are both fruits by reading massive amounts of Internet text, what you can do is then take this word embedding and apply it to your named entity recognition task, for which you might have a much smaller training set, maybe just 100,000 words in your training set, or even much smaller.

And so this allows you to carry out transfer learning, where you take information you've learned from huge amounts of unlabeled text that you can suck down essentially for free off the Internet to figure out that orange, apple, and durian are fruits.

And then transfer that knowledge to a task, such as named entity recognition, for which you may have a relatively small labeled training set. And, of course, for simplicity, l drew this for it only as a unidirectional RNN. If you actually want to carry out the named entity recognition task, you should, of course, use a bidirectional RNN rather than a simpler one I've drawn here.

![image](https://user-images.githubusercontent.com/60442877/167545856-6541ee5a-0cc1-4777-8532-fcaa22cdfb9e.png)

But to summarize, this is how you can carry out transfer learning using word embeddings. Step 1 is to learn word embeddings from a large text corpus, a very large text corpus or you can also download pre-trained word embeddings online. There are several word embeddings that you can find online under very permissive licenses. And you can then take these word embeddings and transfer the embedding to new task, where you have a much smaller labeled training sets. And use this, let's say, 300 dimensional embedding, to represent your words. One nice thing also about this is you can now use relatively lower dimensional feature vectors. So rather than using a 10,000 dimensional one-hot vector, you can now instead use maybe a 300 dimensional dense vector. Although the one-hot vector is fast and the 300 dimensional vector that you might learn for your embedding will be a dense vector. And then, finally, as you train your model on your new task, on your named entity recognition task with a smaller label data set, one thing you can optionally do is to continue to fine tune, continue to adjust the word embeddings with the new data. In practice, you would do this only if this task has a pretty big data set. If your label data set for step 2 is quite small, then usually, I would not bother to continue to fine tune the word embeddings. So word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller training set. So it has been useful for many NLP tasks. It has been less useful for language modeling, machine translation, especially if you're accessing a language modeling or machine translation task for which you have a lot of data just dedicated to that task. So as seen in other transfer learning settings, if you're trying to transfer from some task A to some task B, the process of transfer learning is just most useful when you happen to have a ton of data for A and a relatively smaller data set for B. And so that's true for a lot of NLP tasks, and just less true for some language modeling and machine translation settings.

# Face Encoding vs Word Embedding

Finally, word embeddings has a interesting relationship to the face encoding ideas that you learned about in the previous course, if you took the convolutional neural networks course. So you will remember that for face recognition, we train this Siamese network architecture that would learn, say, a 128 dimensional representation for different faces. And then you can compare these encodings in order to figure out if these two pictures are of the same face. The words encoding and embedding mean fairly similar things. 

One difference between the face recognition literature and what we do in word embeddings is that, for face recognition, you wanted to train a neural network that can take as input any face picture, even a picture you've never seen before, and have a neural network compute an encoding for that new picture. Whereas what we'll do, and you'll understand this better when we go through the next few videos, whereas what we'll do for learning word embeddings is that we'll have a fixed vocabulary of, say, 10,000 words. And we'll learn a vector e1 through, say, e10,000 that just learns a fixed encoding or learns a fixed embedding for each of the words in our vocabulary. So that's one difference between the set of ideas you saw for face recognition versus what the algorithms we'll discuss in the next few videos. But the terms encoding and embedding are used somewhat interchangeably. So the difference I just described is not represented by the difference in terminologies. It's just a difference in how we need to use these algorithms in face recognition, where there's unlimited sea of pictures you could see in the future. Versus natural language processing, where there might be just a fixed vocabulary, and everything else like that we'll just declare as an unknown word.

Overall, by replacing the one-hot vectors we're using previously with the embedding vectors, you can allow your algorithms to generalize much better, or you can learn from much less label data. 

![image](https://user-images.githubusercontent.com/60442877/168404232-a3569c93-fabc-4126-b31d-39afe8778633.png)

## Properties of Word Embeddings

By now, you should have a sense of how word embeddings can help you build NLP applications. One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. And while reasonable analogies may not be by itself the most important NLP application, they might also help convey a sense of what these word embeddings are doing, what these word embeddings can do. 

![image](https://user-images.githubusercontent.com/60442877/168404519-2b2f00d9-3791-44a9-a8ea-ba795c087a5f.png)

Let me show you what I mean here are the featurized representations of a set of words that you might hope a word embedding could capture. Let's say I pose a question, man is to woman as king is to what? Many of you will say, man is to woman as king is to queen. But is it possible to have an algorithm figure this out automatically? Well, here's how you could do it, let's say that you're using this four dimensional vector to represent man. So this will be your E5391, although just for this video, let me call this e subscript man. And let's say that's the embedding vector for woman, so I'm going to call that e subscript woman, and similarly for king and queen. And for this example, I'm just going to assume you're using four dimensional embeddings, rather than anywhere from 50 to 1,000 dimensional, which would be more typical. One interesting property of these vectors is that if you take the vector, e man, and subtract the vector e woman, then, You end up with approximately -1, negative another 1 is -2, decimal 0- 0, 0- 0, close to 0- 0, so you get roughly -2 0 0 0. And similarly if you take e king minus e queen, then that's approximately the same thing. That's about -1- 0.97, it's about -2. This is about 1- 1, since kings and queens are both about equally royal. So that's 0, and then age difference, food difference, 0. And so what this is capturing is that the main difference between man and woman is the gender. And the main difference between king and queen, as represented by these vectors, is also the gender. Which is why the difference e man- e woman, and the difference e king- e queen, are about the same.

![image](https://user-images.githubusercontent.com/60442877/168406411-4de09adb-e935-4d58-94ad-71d7147f05d1.png)

So one way to carry out this analogy reasoning is, if the algorithm is asked, man is to woman as king is to what? What it can do is compute e man- e woman, and try to find a vector, try to find a word so that e man- e woman is close to e king- e of that new word. And it turns out that when queen is the word plugged in here, then the left hand side is close to the the right hand side.

Before moving on, I just want to clarify what this plot on the left is. Previously, we talked about using algorithms like t-SAE to visualize words. What t-SAE does is, it takes 300-D data, and it maps it in a very non-linear way to a 2D space. And so the mapping that t-SAE learns, this is a very complicated and very non-linear mapping. So after the t-SAE mapping, you should not expect these types of parallelogram relationships, like the one we saw on the left, to hold true. And it's really in this original 300 dimensional space that you can more reliably count on these types of parallelogram relationships in analogy pairs to hold true. And it may hold true after a mapping through t-SAE, but in most cases, because of t-SAE's non-linear mapping, you should not count on that. And many of the parallelogram analogy relationships will be broken by t-SAE.

![image](https://user-images.githubusercontent.com/60442877/168409979-1d5ca6e1-a1f1-4ab2-80e0-5729924dc8b1.png)


Now before moving on, let me just describe the similarity function that is most commonly used. So the most commonly used similarity function is called cosine similarity. So this is the equation we had from the previous slide. So in cosine similarity, you define the similarity between two vectors u and v as u transpose v divided by the lengths by the Euclidean lengths. So ignoring the denominator for now, this is basically the inner product between u and v. And so if u and v are very similar, their inner product will tend to be large. And this is called cosine similarity because this is actually the cosine of the angle between the two vectors, u and v. So that's the angle phi, so this formula is actually the cosine of the angle between them.

So in this video, you saw how word embeddings can be used for analogy reasoning. And while you might not be trying to build an analogy reasoning system yourself as an application, this I hope conveys some intuition about the types of feature-like representations that these representations can learn. And you also saw how cosine similarity can be a way to measure the similarity between two different word embeddings. Now, we talked a lot about properties of these embeddings and how you can use them. Next, let's talk about how you'd actually learn these word embeddings, let's go on to the next video.
