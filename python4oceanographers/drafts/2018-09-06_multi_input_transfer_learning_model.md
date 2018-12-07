Title: Multi-input Transfer Learning Model - One model to rule them all
date:  2018-09-06 04:43
comments: true
slug: multi_input_transfer_learning
Category: Python, kaggle, coursera,
Tags: deeplearning, pretrained models
description: This post particularly provides an advanced way to use pretrained neural nets for classifying images.
Keywords: deeplearning, pretrained models, use pretrained neural nets for classifying images, deep learning, kaggle competition dogs vs cats, neural neworks pretrained
Keywords: kaggle, competition, deep learning, kaggle tricks, modelling, transfer learning

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/lotr_rules.png"  height="1000" width="700" ></center>
</div>

So in my day to day I am working on a image classification task. Now I won't go much into explaining transfer learning here. Those of you that want to understand it can read it at my blog [here](http://mlwhiz.com/blog/2017/04/17/deep_learning_pretrained_models/).

Anyways this post is intended for folks who understand how transfer learning works and are trying to squeeze out some extra performance from the same. Till now we normally do something like this:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/before_ensemble.png"  height="800" width="600" ></center>
</div>

For example: Lets say you are working for a kaggle competition and you create multiple models for the image classification task using precomputed features from all these different pretrained models. You might then stack these different models or create a mean ensemble out of these models. And really that is a pretty good idea. I myself went with this idea for a lot of stuff not long back.

Than an epiphany stuck me. Why not let the neural network itself learn the weights while training the additional layers itself. That would be pretty cool. So i tried the same. So, why don't we do this:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/now_lotr.png"  height="800" width="600" ></center>
</div>

In my experiments I have noticed that it works great.

## Why it works?

Why it works? This approach might look similar to an ensemble approach, but it is not that exactly since the weights are getting trained on all the layers at once. From an understanding point of view, I believe that when we give input from different architectures , the network is able to extract value using interactions between different architectures on its own.

## Code:

Some code to do this:

<pre style="font-size:80%; padding:7px; margin:0em;">
<code class="python"># create input Layers for precomputed feats from different pretrained models
Xception_i = Input(shape=train_precomputed_Xception.shape[1:],  name='Xception')
ResNet50_i = Input(shape=train_precomputed_ResNet50.shape[1:],  name='ResNet50')
InceptionV3_i = Input(shape=train_precomputed_InceptionV3.shape[1:],name='InceptionV3')
InceptionResNetV2_i = Input(shape=train_precomputed_InceptionResNetV2.shape[1:],name='InceptionResNetV2')
DenseNet201_i = Input(shape=train_precomputed_DenseNet201.shape[1:],  name='DenseNet201')
DenseNet169_i = Input(shape=train_precomputed_DenseNet169.shape[1:],  name='DenseNet169')
VGG19_i = Input(shape=train_precomputed_VGG19.shape[1:],  name='VGG19')

# Add additional layers on top of each input layer
last_layers = [Xception_i,ResNet50_i,InceptionV3_i,InceptionResNetV2_i,DenseNet201_i,DenseNet169_i,VGG19_i]
last_layers_name = ['Xception_3d','ResNet50_3d','InceptionV3_3d','InceptionResNetV2_3d','DenseNet201_3d','DenseNet169_3d','VGG19_3d']
new_last_layers = {}
for i, last_layer in enumerate(last_layers):
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    new_last_layers[last_layers_name[i]] = x

# Concatenate the output of the additional layers
x = concatenate([new_last_layers['Xception_3d'],new_last_layers['ResNet50_3d'],new_last_layers['InceptionV3_3d'],new_last_layers['InceptionResNetV2_3d'],new_last_layers['DenseNet201_3d'],new_last_layers['DenseNet169_3d'],new_last_layers['VGG19_3d']])

# Add a softmax layer
main_output = Dense(nb_classes, activation='softmax', name='main_output')(x)

# create the model using Keras Functional API
model = Model(inputs=[Xception_i,ResNet50_i,InceptionV3_i,InceptionResNetV2_i,DenseNet201_i,DenseNet169_i,VGG19_i], outputs=[main_output])

# compile
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics =["accuracy"])

# train - Use any callbacks you deem necessary
model.fit([train_precomputed_Xception,train_precomputed_ResNet50,train_precomputed_InceptionV3,train_precomputed_InceptionResNetV2,train_precomputed_DenseNet201,train_precomputed_DenseNet169,train_precomputed_VGG19], onehot_train_y, batch_size=128, nb_epoch=10, validation_data=([valid_precomputed_Xception,valid_precomputed_ResNet50,valid_precomputed_InceptionV3,valid_precomputed_InceptionResNetV2,valid_precomputed_DenseNet201,valid_precomputed_DenseNet169,valid_precomputed_VGG19] ,onehot_valid_y),shuffle=True)

</code></pre>

And we are done!

Let me know any other ideas regarding the same. Or how you liked the idea.
