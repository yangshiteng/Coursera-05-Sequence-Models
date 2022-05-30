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

## Face Encoding vs Word Embedding

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


## Embedding Matrix

Let's start to formalize the problem of learning a good word embedding. When you implement an algorithm to learn a word embedding, what you end up learning is an embedding matrix.

![image](https://user-images.githubusercontent.com/60442877/168410300-63c04fac-13f6-4c77-8f8f-2e195a326d79.png)

Embedding matrix times the one-hot vector results in the single column of the selected word in Embedding matrix


## Learning Word Embeddings

In this video, you'll start to learn some concrete algorithms for learning word embeddings. In the history of deep learning as applied to learning word embeddings, people actually started off with relatively complex algorithms. And then over time, researchers discovered they can use simpler and simpler and simpler algorithms and still get very good results especially for a large dataset. But what happened is, some of the algorithms that are most popular today, they are so simple that if I present them first, it might seem almost a little bit magical, how can something this simple work? So, what I'm going to do is start off with some of the slightly more complex algorithms because I think it's actually easier to develop intuition about why they should work, and then we'll move on to simplify these algorithms and show you some of the simple algorithms that also give very good results.

![image](https://user-images.githubusercontent.com/60442877/168441904-2a6e6a65-ab17-4684-a410-4ebb6d558106.png)

Let's say you're building a language model and you do it with a neural network. So, during training, you might want your neural network to do something like input, I want a glass of orange, and then predict the next word in the sequence. And below each of these words, I have also written down the index in the vocabulary of the different words. So it turns out that building a neural language model is a reasonable way to learn a set of embeddings. 

So this is going to be 10,000 dimensional vector. And what we're going to do is then have a matrix of parameters E, and take E times O to get an embedding vector e4343, and this step really means that e4343 is obtained by the matrix E times the one-hot vector 43. So now you have a bunch of 300 dimensional embedding vector. And what we can do, is fill all of them into a neural network. So here is the neural network layer. And then this neural network feeds to a softmax, which has it's own parameters as well. And a softmax classifies among the 10,000 possible outputs in the vocab for those final word we're trying to predict. And so, if in the training dataset we saw the word juice then, the target for the softmax during training will be that it should predict the other word juice was what came after this.

So this hidden layer here will have his own parameters. So have some, I'm going to call this W1 and there's also B1. The softmax there was this own parameters W2, B2, and they're using 300 dimensional word embeddings, then here we have six words. So, this would be six times 300. So this layer or this input will be a 1,800 dimensional vector obtained by taking your six embedding vectors and stacking them together. Well, what's actually more commonly done is to have a fixed historical window. So for example, you might decide that you always want to predict the next word given say the previous four words, where four here is a hyperparameter of the algorithm. So this is how you adjust to either very long or very short sentences or you decide to always just look at the previous four words, so you say, I will still use those four words. And so, let's just get rid of these. And so, if you're always using a four word history, this means that your neural network will input a 1,200 dimensional feature vector, go into this layer, then have a softmax and try to predict the output. And again, variety of choices. And using a fixed history, just means that you can deal with even arbitrarily long sentences because the input sizes are always fixed. So, the parameters of this model will be this matrix E, and use the same matrix E for all the words. So you don't have different matrices for different positions in the proceedings four words, is the same matrix E. And then, these weights are also parameters of the algorithm and you can use that backprop to perform gradient descent to maximize the likelihood of your training set to just repeatedly predict given four words in a sequence, what is the next word in your text corpus? 

So, this is one of the earlier and pretty successful algorithms for learning word embeddings, for learning this matrix E. But now let's generalize this algorithm and see how we can derive even simpler algorithms.

![image](https://user-images.githubusercontent.com/60442877/168442242-38b2d61c-61fa-4bb0-a52f-258eb518867b.png)

To summarize, in this video you saw how the language modeling problem which causes the pose of machines learning problem where you input the context like the last four words and predicts some target words, how posing that problem allows you to learn input word embedding. In the next video, you'll see how using even simpler context and even simpler learning algorithms to map from context to target word, can also allow you to learn a good word embedding. Let's go on to the next video where we'll discuss the Word2Vec model.

## Word2Vec （Skip-gram model）

In the last video, you saw how you can learn a neural language model in order to get good word embeddings. In this video, you see the Word2Vec algorithm which is simpler and computationally more efficient way to learn this types of embeddings

![image](https://user-images.githubusercontent.com/60442877/168455482-2cdcc34e-ec09-4ae1-892b-b6fb20b0d204.png)

Let's say you're given this sentence in your training set. In the skip-gram model, what we're going to do is come up with a few context to target pairs to create our supervised learning problem. So rather than having the context be always the last four words or the last end words immediately before the target word, what I'm going to do is, say, randomly pick a word to be the context word. And let's say we chose the word orange. And what we're going to do is randomly pick another word within some window. Say plus minus five words or plus minus ten words of the context word and we choose that to be target word. So maybe just by chance you might pick juice to be a target word, that's just one word later. Or you might choose two words before. So you have another pair where the target could be glass or, Maybe just by chance you choose the word my as the target. And so we'll set up a supervised learning problem where given the context word, you're asked to predict what is a randomly chosen word within say, a plus minus ten word window, or plus minus five or ten word window of that input context word. And obviously, this is not a very easy learning problem, because within plus minus 10 words of the word orange, it could be a lot of different words. But a goal that's setting up this supervised learning problem, isn't to do well on the supervised learning problem per se, it is that we want to use this learning problem to learn good word embeddings

![image](https://user-images.githubusercontent.com/60442877/168455870-85a6f135-4586-4f06-ac77-f8abe9ede78d.png)

So to summarize, this is the overall little model, little neural network with basically looking up the embedding and then just a soft max unit. And the matrix E will have a lot of parameters, so the matrix E has parameters corresponding to all of these embedding vectors, E subscript C. And then the softmax unit also has parameters that gives the theta T parameters but if you optimize this loss function with respect to the all of these parameters, you actually get a pretty good set of embedding vectors. So this is called the skip-gram model because is taking as input one word like orange and then trying to predict some words skipping a few words from the left or the right side.

![image](https://user-images.githubusercontent.com/60442877/168456123-afe0b2bc-9c67-4aeb-94fd-79e474aa4951.png)

Now, it turns out there are a couple problems with using this algorithm. And the primary problem is computational speed. In particular, for the softmax model, every time you want to evaluate this probability, you need to carry out a sum over all 10,000 words in your vocabulary. And maybe 10,000 isn't too bad, but if you're using a vocabulary of size 100,000 or a 1,000,000, it gets really slow to sum up over this denominator every single time. And, in fact, 10,000 is actually already that will be quite slow, but it makes even harder to scale to larger vocabularies. 

So there are a few solutions to this, one which you see in the literature is to use a hierarchical softmax classifier. And what that means is, instead of trying to categorize something into all 10,000 carries on one go. Imagine if you have one classifier, it tells you is the target word in the first 5,000 words in the vocabulary? Or is in the second 5,000 words in the vocabulary? And let's say this binary cost that it tells you this is in the first 5,000 words, think of second class to tell you that this in the first 2,500 words of vocab or in the second 2,500 words vocab and so on. Until eventually you get down to classify exactly what word it is, so that the leaf of this tree, and so having a tree of classifiers like this, means that each of the retriever nodes of the tree can be just a binding classifier. And so you don't need to sum over all 10,000 words or else it will capsize in order to make a single classification. In fact, the computational classifying tree like this scales like log of the vocab size rather than linear in vocab size. So this is called a hierarchical softmax classifier.

## Negative Embeddings (Skip-gram model, deal with the computational problem in softmax)

In the last video, you saw how the Skip-Gram model allows you to construct a supervised learning task. So we map from context to target words and how that allows you to learn a useful word embedding. But the downside of that was the Softmax objective was slow to compute. In this video, you'll see a modified learning problem called negative sampling that allows you to do something similar to the Skip-Gram model you saw just now, but with a much more efficient learning algorithm.
 
![image](https://user-images.githubusercontent.com/60442877/168478144-2cf63536-d461-4021-a5c6-40c7cdee2dd5.png)

So what we're going to do in this algorithm is create a new supervised learning problem. And the problem is, given a pair of words like orange and juice, we're going to predict, is this a context-target pair? So in this example, orange juice was a positive example. And how about orange and king? Well, that's a negative example, so I'm going to write 0 for the target. So to summarize, the way we generated this data set is, we'll pick a context word and then pick a target word and that is the first row of this table. That gives us a positive example. So context, target, and then give that a label of 1. And then what we'll do is for some number of times say, k times, we're going to take the same context word and then pick random words from the dictionary, king, book, the, of, whatever comes out at random from the dictionary and label all those 0, and those will be our negative examples.

Then we're going to create a supervised learning problem where the learning algorithm inputs x, inputs this pair of words, and it has to predict the target label to predict the output y.

How do you choose k, Mikolov et al, recommend that maybe k is 5 to 20 for smaller data sets. And if you have a very large data set, then chose k to be smaller. So k equals 2 to 5 for larger data sets, and large values of k for smaller data sets. 

![image](https://user-images.githubusercontent.com/60442877/168479535-38492695-111e-4cad-97b4-3ecc6e3854e6.png)

Next, let's describe the supervised learning model for learning a mapping from x to y. So what we're going to do is define a logistic regression model. Say, that the chance of y = 1, given the input c, t pair, we're going to model this as basically a regression model, but the specific formula we'll use sigmoid applied to theta t transpose, e c. So the parameters are similar as before, you have one parameter vector theta for each possible target word. And a separate parameter vector, really the embedding vector, for each possible context word. And we're going to use this formula to estimate the probability that y is equal to 1. So if you have k examples here, then you can think of this as having a k to 1 ratio of negative to positive examples. So for every positive examples, you have k negative examples with which to train this logistic regression-like model.

And so to draw this as a neural network, if the input word is orange, which is word 6257, then what you do is, you input the one-hot vector passing through e, do the multiplication to get the embedding vector 6257. And then what you have is really 10,000 possible logistic regression classification problems. Where one of these will be the classifier corresponding to, well, is the target word juice or not? And then there will be other words, for example, there might be ones somewhere down here which is predicting, is the word king or not and so on, for these possible words in your vocabulary. So think of this as having 10,000 binary logistic regression classifiers, but instead of training all 10,000 of them on every iteration, we're only going to train five of them. We're going to train the one responding to the actual target word we got and then train four randomly chosen negative examples. And this is for the case where k is equal to 4. So instead of having one giant 10,000 way Softmax, which is very expensive to compute, we've instead turned it into 10,000 binary classification problems, each of which is quite cheap to compute. And on every iteration, we're only going to train five of them or more generally, k + 1 of them, of k negative examples and one positive examples. And this is why the computation cost of this algorithm is much lower because you're updating k + 1, let's just say units, k + 1 binary classification problems. Which is relatively cheap to do on every iteration rather than updating a 10,000 way Softmax classifier.

So this technique is called negative sampling because what you're doing is, you have a positive example, the orange and then juice. And then you will go and deliberately generate a bunch of negative examples, negative samplings, hence, the name negative sampling, with which to train four more of these binary classifiers. And on every iteration, you choose four different random negative words with which to train your algorithm on.

So to summarize, you've seen how you can learn word vectors in a Softmax classier, but it's very computationally expensive. And in this video, you saw how by changing that to a bunch of binary classification problems, you can very efficiently learn words vectors. And if you run this algorithm, you will be able to learn pretty good word vectors. Now of course, as is the case in other areas of deep learning as well, there are open source implementations. And there are also pre-trained word vectors that others have trained and released online under permissive licenses. And so if you want to get going quickly on a NLP problem, it'd be reasonable to download someone else's word vectors and use that as a starting point. So that's it for the Skip-Gram model. In the next video, I want to share with you yet another version of a word embedding learning algorithm that is maybe even simpler than what you've seen so far. So in the next video, let's learn about the Glove algorithm.

## GloVE Word Vectors

You learn about several algorithms for computing words embeddings. Another algorithm that has some momentum in the NLP community is the GloVe algorithm. This is not used as much as the Word2Vec or the skip-gram models, but it has some enthusiasts. Because I think, in part of its simplicity.

![image](https://user-images.githubusercontent.com/60442877/168493547-891c58ce-d2b0-43be-9e2b-905c4a4c4607.png)

![image](https://user-images.githubusercontent.com/60442877/168493684-af390ae5-7180-4835-9bdf-ebaaa7624c92.png)

## Applications Using Word Embeddings

### Sentiment Classification

Sentiment classification is the task of looking at a piece of text and telling if someone likes or dislikes the thing they're talking about. It is one of the most important building blocks in NLP and is used in many applications. One of the challenges of sentiment classification is you might not have a huge label training set for it. But with word embeddings, you're able to build good sentiment classifiers even with only modest-size label training sets. 

![image](https://user-images.githubusercontent.com/60442877/168511705-aa90f4a6-29ee-42d2-b993-c21a0132d6a1.png)

Let's see how you can do that. So here's an example of a sentiment classification problem. The input X is a piece of text and the output Y that you want to predict is what is the sentiment, such as the star rating of, let's say, a restaurant review. So if someone says, "The dessert is excellent" and they give it a four-star review, "Service was quite slow" two-star review, "Good for a quick meal but nothing special" three-star review. So if you can train a system to map from X or Y based on a label data set like this, then you could use it to monitor comments that people are saying about maybe a restaurant that you run.

So one of the challenges of sentiment classification is you might not have a huge label data set. So for sentimental classification task, training sets with maybe anywhere from 10,000 to maybe 100,000 words would not be uncommon. Sometimes even smaller than 10,000 words and word embeddings that you can take can help you to much better understand especially when you have a small training set. 

![image](https://user-images.githubusercontent.com/60442877/168512197-422e2db0-083f-45c9-af18-f18c058a98af.png)

Here's a simple sentiment classification model. You can take a sentence like "dessert is excellent" and look up those words in your dictionary. We use a 10,000-word dictionary as usual. And let's build a classifier to map it to the output Y that this was four stars. So given these four words, as usual, we can take these four words and look up the one-hot vector. So there's 0 8 9 2 8 which is a one-hot vector multiplied by the embedding matrix E, which can learn from a much larger text corpus. It can learn in embedding from, say, a billion words or a hundred billion words, and use that to extract out the embedding vector for the word "the", and then do the same for "dessert", do the same for "is" and do the same for "excellent". And if this was trained on a very large data set, like a hundred billion words, then this allows you to take a lot of knowledge even from infrequent words and apply them to your problem, even words that weren't in your labeled training set. 

Now here's one way to build a classifier, which is that you can take these vectors, let's say these are 300-dimensional vectors, and you could then just sum or average them. And I'm just going to put a bigger average operator here and you could use sum or average. And this gives you a 300-dimensional feature vector that you then pass to a soft-max classifier which then outputs Y-hat. And so the softmax can output what are the probabilities of the five possible outcomes from one-star up to five-star.

So one of the problems with this algorithm is it ignores word order. In particular, this is a very negative review, "Completely lacking in good taste, good service, and good ambiance". But the word good appears a lot. This is a lot. Good, good, good. So if you use an algorithm like this that ignores word order and just sums or averages all of the embeddings for the different words, then you end up having a lot of the representation of good in your final feature vector and your classifier will probably think this is a good review even though this is actually very harsh. This is a one-star review. So here's a more sophisticated model which is that, instead of just summing all of your word embeddings, you can instead use a RNN for sentiment classification.

![image](https://user-images.githubusercontent.com/60442877/168513678-d73e7c2d-f859-4651-9abe-c89601362405.png)

So here's what you can do. You can take that review, "Completely lacking in good taste, good service, and good ambiance", and find for each of them, the one-hot vector. And so I'm going to just skip the one-hot vector representation but take the one-hot vectors, multiply it by the embedding matrix E as usual, then this gives you the embedding vectors and then you can feed these into an RNN. And the job of the RNN is to then compute the representation at the last time step that allows you to predict Y-hat. So this is an example of a many-to-one RNN architecture which we saw in the previous week. 

And with an algorithm like this, it will be much better at taking word sequence into account and realize that "things are lacking in good taste" is a negative review and "not good" a negative review unlike the previous algorithm, which just sums everything together into a big-word vector mush and doesn't realize that "not good" has a very different meaning than the words "good" or "lacking in good taste" and so on. And so if you train this algorithm, you end up with a pretty decent sentiment classification algorithm and because your word embeddings can be trained from a much larger data set, this will do a better job generalizing to maybe even new words now that you'll see in your training set, such as if someone else says, "Completely absent of good taste, good service, and good ambiance" or something, then even if the word "absent" is not in your label training set, if it was in your 1 billion or 100 billion word corpus used to train the word embeddings, it might still get this right and generalize much better even to words that were in the training set used to train the word embeddings but not necessarily in the label training set that you had for specifically the sentiment classification problem. So that's it for sentiment classification, and I hope this gives you a sense of how once you've learned or downloaded from online a word embedding, this allows you to quite quickly build pretty effective NLP systems.

### Debiasing Word Embeddings

![image](https://user-images.githubusercontent.com/60442877/168515598-20ec8b81-29d0-4e26-82da-e68b9baa8e75.png)

![image](https://user-images.githubusercontent.com/60442877/168518605-969bce41-63ce-4332-8d56-2609c4343e8d.png)

So to summarize, I think that reducing or eliminating bias of our learning algorithms is a very important problem because these algorithms are being asked to help with or to make more and more important decisions in society. In this video I shared just one set of ideas for how to go about trying to address this problem, but this is still a very much an ongoing area of active research by many researchers. 


# Week 3 - Sequence to sequence model

## Basic Models

### Language translation

In this week, you'll hear about sequence to sequence models, which are useful for everything from machine translation to speech recognition. Let's start with the basic models, and then later this week, you'll hear about beam surge, the attention model, and we will wrap up the discussion of models for audio data like speech.

Let's say you want to input a French sentence like Jane visite I'Afrique Septembre, and you want to translate it to the English sentence, Jane is visiting Africa in September. As usual, let's use x1 through x, in this case 5 to represent the words and the input sequence, and we'll use y1 through y6 to represent the words in the output sequence. How can you train a neural network to input the sequence x and output the sequence y?

![image](https://user-images.githubusercontent.com/60442877/170882730-6d6f9889-a1c4-4fc2-a1f2-5da6977b49db.png)

One of the most remarkable recent results in deep learning is that this model works. Given enough pairs of French and English sentences, if you train a model to input a French sentence and output the corresponding English translation, this will actually work decently well. This model simply uses an encoding network whose job is to find an encoding of the input French sentence, and then use a decoding network to generate the corresponding English translation.

### Image Captioning

![image](https://user-images.githubusercontent.com/60442877/170883462-2530c707-2c73-45c3-a771-ad7796b6f7e8.png)

An architecture very similar to this also works for image captioning. Given an image like the one shown here, maybe you wanted to be captions automatically as a cat sitting on a chair. How do you train in your network to input an image and output a caption like that face up there? Here's what you can do.

From the earlier course on ConvNet, you've seen how you can input an image into a convolutional network, maybe a pre-trained AlexNet, and have that learn and encoding a learner set the features of the input image. So, This is actually the AlexNet architecture, and if we get rid of this final softmax unit, the pre-trained AlexNet can give you a 4,096-dimensional feature vector of which to represent this picture of a cat. This pre-trained network can be the encoded network for the image and you now have a 4,096-dimensional vector that represents the image. You can then take this and feed it to an RNN whose job it is to generate the caption one word at a time. Similar to what we saw with machine translation, translating from French the English, you can now input a feature vector describing the inputs and then have it generate an output set of words, one word at a time. This actually works pretty well for image captioning, especially if the caption you want to generate is not too long.

You've now seen how a basic sequence to sequence model works. How basic image to sequence, or image captioning model works. But there are some differences between how you'll run a model like this, to generate the sequence compared to how you were synthesizing novel text using a language model. One of the key differences is you don't want to randomly choose in translation. You may be want the most likely translation or you don't want to randomly choose in caption, maybe not, but you might want the best caption and most likely caption. Let's see in the next video how you go about generating that.

## Picking the Most Likely Sentence

There are some similarities between the sequence to sequence machine translation model and the language models that you have worked within the first week of this course, but there are some significant differences as well.

![image](https://user-images.githubusercontent.com/60442877/170883815-1ef30c84-cc8d-4fac-8a75-a89d1ba6bef0.png)

Let's take a look. So, you can think of machine translation as building a conditional language model. Here's what I mean, in language modeling, this was the network we had built in the first week. And this model allows you to estimate the probability of a sentence. That's what a language model does. And you can also use this to generate novel sentences, and sometimes when you are writing x1 and x2 here, where in this example, x2 would be equal to y1 or equal to y and one is just a feedback. But x1, x2, and so on were not important. So just to clean this up for this slide, I'm going to just cross these off. X1 could be the vector of all zeros and x2, x3 are just the previous output you are generating. So that was the language model. The machine translation model looks as follows, and I am going to use a couple different colors, green and purple, to denote respectively the encoded network in green and the decoded network in purple. And you notice that the decoded network looks pretty much identical to the language model that we had up there. So what the machine translation model is, is very similar to the language model, except that instead of always starting along with the vector of all zeros, it instead has an encoded network that figures out some representation for the input sentence, and it takes that input sentence and starts off the decoded network with representation of the input sentence rather than with the representation of all zeros. So, that's why I call this a conditional language model, and instead of modeling the probability of any sentence, it is now modeling the probability of, say, the output English translation, conditions on some input French sentence. So in other words, you're trying to estimate the probability of an English translation. Like, what's the chance that the translation is "Jane is visiting Africa in September," but conditions on the input French censors like, "Jane visite I'Afrique en septembre." So, this is really the probability of an English sentence conditions on an input French sentence which is why it is a conditional language model. 

![image](https://user-images.githubusercontent.com/60442877/170884183-e668dd3d-64cc-41f1-9e0a-2c5bbb22a871.png)

So, when you're using this model for machine translation, you're not trying to sample at random from this distribution. Instead, what you would like is to find the English sentence, y, that maximizes that conditional probability. So in developing a machine translation system, one of the things you need to do is come up with an algorithm that can actually find the value of y that maximizes this term over here. The most common algorithm for doing this is called beam search, and it's something you'll see in the next video. 

Why not use a greedy search?

So, what is greedy search? Well, greedy search is an algorithm from computer science which says to generate the first word just pick whatever is the most likely first word according to your conditional language model. Going to your machine translation model and then after having picked the first word, you then pick whatever is the second word that seems most likely, then pick the third word that seems most likely. This algorithm is called greedy search.

![image](https://user-images.githubusercontent.com/60442877/170886145-cc2d9265-f5dc-4779-a183-077ed753add7.png)

So, to summarize, in this video, you saw how machine translation can be posed as a conditional language modeling problem. But one major difference between this and the earlier language modeling problems is rather than wanting to generate a sentence at random, you may want to try to find the most likely English sentence, most likely English translation. But the set of all English sentences of a certain length is too large to exhaustively enumerate. So, we have to resort to a search algorithm. So, with that, let's go onto the next video where you'll learn about beam search algorithm.

## Beam Search

In this video, you learn about the beam search algorithm. In the last video, you remember how for machine translation given an input French sentence, you don't want to output a random English translation, you want to output the best and the most likely English translation. The same is also true for speech recognition where given an input audio clip, you don't want to output a random text transcript of that audio, you want to output the best, maybe the most likely, text transcript. Beam search is the most widely used algorithm to do this.

![image](https://user-images.githubusercontent.com/60442877/170889444-6ba93ae8-8bd3-4cf0-bb77-04cff2c109c1.png)

Let's just try Beam Search using our running example of the French sentence, "Jane, visite l'Afrique en Septembre". Hopefully being translated into, "Jane, visits Africa in September". The first thing Beam search has to do is try to pick the first words of the English translation, that's going to output. So here I've listed, say, 10,000 words into vocabulary. And to simplify the problem a bit, I'm going to ignore capitalization. So I'm just listing all the words in lower case. So, in the first step of Beam Search, I use this network fragment with the coalition in green and decoalition in purple, to try to evaluate what is the probability of that first word. So, what's the probability of the first output y, given the input sentence x gives the French input. So, whereas greedy search will pick only the one most likely words and move on, Beam Search instead can consider multiple alternatives. So, the Beam Search algorithm has a parameter called B, which is called the beam width and for this example I'm going to set the beam width to be equal to the three. And what this means is Beam search will consider not just one possibility but consider three at the time. So in particular, let's say evaluating this probability over different choices the first words, it finds that the choices in, Jane and September are the most likely three possibilities for the first words in the English outputs.

So, to be clear in order to perform this first step of Beam search, what you need to do is run the input French sentence through this encoder network and then this first step will then decode the network, this is a softmax output overall 10,000 possibilities. Then you would take those 10,000 possible outputs and keep in memory which were the top three.

Let's go into the second step of Beam search. 

![image](https://user-images.githubusercontent.com/60442877/170889752-cb0170ae-8c05-412e-8331-c6bde925fab1.png)

So for this second step of beam search because we're continuing to use a beam width of three and because there are 10,000 words in the vocabulary you'd end up considering three times 10000 or thirty thousand possibilities because there are 10,000 here, 10,000 here, 10,000 here as the beam width times the number of words in the vocabulary and what you do is you evaluate all of these 30000 options according to the probability the first and second words and then pick the top three. Just want to notice that because of beam width is equal to three, every step you instantiate three copies of the network to evaluate these partial sentence fragments and the output. And it's because of beam width is equal to three that you have three copies of the network with different choices for the first words, but these three copies of the network can be very efficiently used to evaluate all 30,000 options for the second word. 

Let's just quickly illustrate one more step of beam search. 

![image](https://user-images.githubusercontent.com/60442877/170889971-1be112a9-c2e3-4dd4-8f24-b3fe69cf8e26.png)

Notice that if the beam width was said to be equal to one, say cause there's only one, then this essentially becomes the greedy search algorithm which we had discussed in the last video but by considering multiple possibilities say three or ten or some other number at the same time beam search will usually find a much better output sentence than greedy search.

## Refinements to Beam Search

In the last video, you saw the basic beam search algorithm, in this video, you learn some little changes, they'll make it work even better.

### Length Normalization

![image](https://user-images.githubusercontent.com/60442877/170891467-4bba2984-6645-4d5e-942e-ae088751f762.png)

### How to choose the Beam width? 

![image](https://user-images.githubusercontent.com/60442877/170891554-85462126-190e-4052-b351-1322bbd9c49e.png)

Share the pros and cons of setting beam to be very large versus very small. If the beam width is very large, then you consider a lot of possibilities and so you tend to get a better result because you're consuming a lot of different options, but it will be slower. The memory requirements will also grow and also be computationally slower. Whereas if you use a very small beam width, then you get a worse result because you are just keeping less possibilities in mind as the algorithm is running, but you get a result faster and the memory requirements will also be lower. In the previous video, we use in our running example a beam width of 3, so we're keeping three possibilities in mind in practice that is on the small side in production systems, it's not uncommon to see a beam width maybe around 10. I think a beam width of 100 would be considered very large for a production system, depending on the application. But for systems where people want to squeeze out every last drop of performance in order to publish people the best possible result, it's not uncommon to see people use beam width of 1,000 or 3,000, but this is very application as well as a domain dependent. I would say try out a variety of values of beam as see what works for your application, but when beam is very large, there is often diminishing returns. For many applications, I would expect to see a huge gain as you go from beam of one, which is basically research to three to maybe 10, but the gains as you go from the thousands of thousand beam width might not be as big.

## Error Analysis in Beam Search

In the third course of this sequence of five courses, you saw how error analysis can help you focus your time on doing the most useful work for your project. Now, beam search is an approximate search algorithm, also called a heuristic search algorithm. And so it doesn't always output the most likely sentence. It's only keeping track of B equals 3 or 10 or 100 top possibilities. So what if beam search makes a mistake? In this video, you'll learn how error analysis interacts with beam search and how you can figure out whether it is the beam search algorithm that's causing problems and worth spending time on. Or whether it might be your RNN model that is causing problems and worth spending time on.

![image](https://user-images.githubusercontent.com/60442877/170899730-794ed211-8d79-4e9d-a0ea-9d383d3766a0.png)

![image](https://user-images.githubusercontent.com/60442877/170900205-e99e98cc-d086-4e77-ba7e-dea4b4af46db.png)

![image](https://user-images.githubusercontent.com/60442877/170900581-e44c9911-e3d5-43fd-bb73-659b32383492.png)


## Bleu Score

One of the challenges of machine translation is that, given a French sentence, there could be multiple English translations that are equally good translations of that French sentence. So how do you evaluate a machine translation system if there are multiple equally good answers, unlike, say, image recognition where there's one right answer? You just measure accuracy. If there are multiple great answers, how do you measure accuracy? The way this is done conventionally is through something called the BLEU score. What the BLEU score does is given a machine generated translation, it allows you to automatically compute a score that measures how good is that machine translation. And the intuition is so long as the machine generated translation is pretty close to any of the references provided by humans, then it will get a high BLEU score. BLEU, by the way, stands for bilingual evaluation understudy.

So, the intuition behind the BLEU score is we're going to look at the machine generated output and see if the types of words it generates appear in at least one of the human generated references. And so these human generated references would be provided as part of the depth set or as part of the test set. 

![image](https://user-images.githubusercontent.com/60442877/170901460-08e75e8f-89b8-4d71-93a6-7c0ed3c534b9.png)

Now, let's look at a somewhat extreme example. Let's say that the machine translation system abbreviating machine translation is MT. So the machine translation, or the MT output, is the the the the the the the. So this is clearly a pretty terrible translation. So one way to measure how good the machine translation output is, is to look at each the words in the output and see if it appears in the references. And so, this would be called a precision of the machine translation output. And in this case, there are seven words in the machine translation output. And every one of these 7 words appears in either Reference 1 or Reference 2, right? So the word the appears in both references. So each of these words looks like a pretty good word to include. So this will have a precision of 7 over 7. It looks like it was a great precision. So this is why the basic precision measure of what fraction of the words in the MT output also appear in the references. This is not a particularly useful measure, because it seems to imply that this MT output has very high precision. So instead, what we're going to use is a modified precision measure in which we will give each word credit only up to the maximum number of times it appears in the reference sentences. So in Reference 1, the word, the, appears twice. In Reference 2, the word, the, appears just once. So 2 is bigger than 1, and so we're going to say that the word, the, gets credit up to twice. So, with a modified precision, we will say that, it gets a score of 2 out of 7, because out of 7 words, we'll give it a 2 credits for appearing.

So here, the denominator is the count of the number of times the word, the, appears of 7 words in total. And the numerator is the count of the number of times the word, the, appears. We clip this count, we take a max, or we clip this count, at 2. So this gives us the modified precision measure.

Now, so far, we've been looking at words in isolation. In the BLEU score, you don't want to just look at isolated words. You maybe want to look at pairs of words as well. Let's define a portion of the BLEU score on bigrams.

![image](https://user-images.githubusercontent.com/60442877/170908356-7ea8e153-3fe3-4280-96b9-d7e759f3d5f2.png)

![image](https://user-images.githubusercontent.com/60442877/170908851-68cb2161-b97f-422a-a09c-c2b6c3b01d25.png)

And one thing that you could probably convince yourself of is if the MT output is exactly the same as either Reference 1 or Reference 2, then all of these values P1, and P2 and so on, they'll all be equal to 1.0. So to get a modified precision of 1.0, you just have to be exactly equal to one of the references. And sometimes it's possible to achieve this even if you aren't exactly the same as any of the references. But you kind of combine them in a way that hopefully still results in a good translation.

![image](https://user-images.githubusercontent.com/60442877/170910322-9220102e-f060-447b-8f13-dd9f8b385961.png)

So the reason the BLEU score was revolutionary for machine translation was because this gave a pretty good, by no means perfect, but pretty good single real number evaluation metric. And so that accelerated the progress of the entire field of machine translation. I hope this video gave you a sense of how the BLEU score works. In practice, few people would implement a BLEU score from scratch. There are open source implementations that you can download and just use to evaluate your own system. But today, BLEU score is used to evaluate many systems that generate text, such as machine translation systems, as well as the example I showed briefly earlier of image captioning systems where you would have a system, have a neural network generated image caption. And then use the BLEU score to see how much that overlaps with maybe a reference caption or multiple reference captions that were generated by people. So the BLEU score is a useful single real number evaluation metric to use whenever you want your algorithm to generate a piece of text. And you want to see whether it has similar meaning as a reference piece of text generated by humans. This is not used for speech recognition, because in speech recognition, there's usually one ground truth. And you just use other measures to see if you got the speech transcription on pretty much, exactly word for word correct. But for things like image captioning, and multiple captions for a picture, it could be about equally good or for machine translations. There are multiple translations, but equally good. The BLEU score gives you a way to evaluate that automatically and therefore speed up your development.

## Attention Model Intuition

For most of this week, you've been using a Encoder-Decoder architecture for machine translation. Where one RNN reads in a sentence and then different one outputs a sentence. There's a modification to this called the Attention Model, that makes all this work much better. The attention algorithm, the attention idea has been one of the most influential ideas in deep learning. Let's take a look at how that works.

Get a very long French sentence like this. What we are asking this green encoder in your network to do is, to read in the whole sentence and then memorize the whole sentences and store it in the activations conveyed here. Then for the purple network, the decoder network till then generate the English translation. Now, the way a human translator would translate this sentence is not to first read the whole French sentence and then memorize the whole thing and then regurgitate an English sentence from scratch. Instead, what the human translator would do is read the first part of it, maybe generate part of the translation. Look at the second part, generate a few more words, look at a few more words, generate a few more words and so on. You kind of work part by part through the sentence, because it's just really difficult to memorize the whole long sentence like that. What you see for the Encoder-Decoder architecture above is that, it works quite well for short sentences, so we might achieve a relatively high Bleu score, but for very long sentences, maybe longer than 30 or 40 words, the performance comes down. Long sentences, it doesn't do well on because it's just difficult to get in your network to memorize a super long sentence.

![image](https://user-images.githubusercontent.com/60442877/171028831-880a96cf-59c2-4a8f-87ee-ae0ef42b1d6d.png)

In this and the next video, you'll see the Attention Model which translates maybe a bit more like humans might, looking at part of the sentence at a time and with an Attention Model, machine translation systems performance can look like this, because by working one part of the sentence at a time, you don't see this huge dip which is really measuring the ability of a neural network to memorize a long sentence which maybe isn't what we most badly need a neural network to do.

![image](https://user-images.githubusercontent.com/60442877/171033743-2f12df3d-0748-40c1-9be9-bdb8d45e3400.png)

Now, the question is, when you're trying to generate this first word, this output, what part of the input French sentence should you be looking at? Seems like you should be looking primarily at this first word, maybe a few other words close by, but you don't need to be looking way at the end of the sentence. What the Attention Model would be computing is a set of attention weights and we're going to use Alpha one, one to denote when you're generating the first words, how much should you be paying attention to this first piece of information here. And then we'll also come up with a second that's called Attention Weight, Alpha one, two which tells us when we're trying to compute the first work of Jane, how much attention we're paying to this second work from the inputs and so on and the Alpha one, three and so on, and together this will tell us what is exactly the context from denoter C that we should be paying attention to, and that is input to this RNN unit to then try to generate the first words.

For the second step of this RNN, we're going to have a new hidden state S two and we're going to have a new set of the attention weights. We're going to have Alpha two, one to tell us when we generate in the second word. I guess this will be visits maybe that being the ground trip label. How much should we paying attention to the first word in the french input and also, Alpha two, two and so on. How much should we paying attention the word visite, how much should we pay attention to the free and so on. And of course, the first word we generate in Jane is also an input to this, and then we have some context that we're paying attention to and the second step, there's also an input and that together will generate the second word and that leads us to the third step, S three, where this is an input and we have some new context C that depends on the various Alpha three for the different time sets, that tells us how much should we be paying attention to the different words from the input French sentence and so on. 

## Detail of Attention Model

In the last video, you saw how the attention model allows a neural network to pay attention to only part of an input sentence while it's generating a translation, much like a human translator might. Let's now formalize that intuition into the exact details of how you would implement an attention model. 

So same as in the previous video, let's assume you have an input sentence and you use a bidirectional RNN, or bidirectional GRU, or bidirectional LSTM to compute features on every word. In practice, GRUs and LSTMs are often used for this, with maybe LSTMs be more common. And so for the forward occurrence, you have a forward occurrence first time step. Activation backward occurrence, first time step. Activation forward occurrence, second time step. Activation backward and so on. For all of them in just a forward fifth time step a backwards fifth time step.

![image](https://user-images.githubusercontent.com/60442877/171036218-dc1dd7b2-5369-4187-9e98-be99847ae916.png)

![image](https://user-images.githubusercontent.com/60442877/171039670-af956b48-8fd8-4b2c-a838-b7028d48307b.png)

Now, one downside to this algorithm is that it does take quadratic time or quadratic cost to run this algorithm. If you have tx words in the input and ty words in the output then the total number of these attention parameters are going to be tx times ty. And so this algorithm runs in quadratic cost. Although in machine translation applications where neither input nor output sentences is usually that long maybe quadratic cost is actually acceptable. 



