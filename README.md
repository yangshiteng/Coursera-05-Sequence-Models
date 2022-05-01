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





