<p align="left"> <img src="/img/logo.png"  width="10"> </p>
# HateBlocker
HateBlocker is an API that utilizes deep learning to detect hateful memes from visual and textual information.


# Problem description
The effect of hateful content on social media can be devastating. A result of a [recent poll](https://www.huffpost.com/entry/social-media-harassment-fake-news-poll-alex-jones_n_5b7b1c53e4b0a5b1febdf30a) has shown that the majority of US adults consider hateful content a serious problem for which social media companies bear the responsibility of detecting and removing. In this project, I used AI to build a hateful meme detector (HateBlocker) and deployed the model as a stand-alone API and a Chrome extension.

# Data
I used data from [Facebook research](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/) which consists of 8.5K training and 0.5 validation examples. This data set consists of image and caption which I run through a couple of pre-trained features extraction models (see below for details). The features are available at [...]. For training, these features should be copied in the [/data](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/data) folder.

# Models
This project has been carried out in collaboration with the research team as [Clarifai](https://www.clarifai.com/), thus the focus was to use the pre-trained models of the Clarifai platform to extract the image and text features. I also performed experiments inspired by new developments in image and video captioning (eg. [VisualBert](https://arxiv.org/abs/1908.03557) and [MDVC](https://arxiv.org/abs/2003.07758))

## [Baseline](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/BaseLine)
The baseline model is a simple logistic regression that is trained on top of the image and caption features from the Clarifai platform. This simple approach gives AUC-ROC=0.63.

## Feature fusion and dense classifier [Concat model](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/Concat)
This classifier is trained on top of several feature sets including image, text, image moderation, text moderation features. The fifth stream of information is supplied by concatenating the top-five concepts of the image to the image caption and then feeding that to the text feature extraction network.

<p align="center"> <img src="/img/model.png"  width="500"> </p>


The best performance of this model (after a lot of fine-tuning) is AUC-ROC = 0.71. Here is also an example that shows how the meaning of a meme changes with modifying either the textual and visual content which highlights the nonlinear interaction of information from image and text.


<p align="center"> <img src="/img/example1.png"  width="500"> </p>


For deployment, this model is coupled with a google vision OCR (for caption extraction) and is implemented as a Flask API. The API inputs a meme URL and outputs the class probabilities. The end-product of this project is a Chrome extension (HateBlocker) which extracts meme URLs and sends them through the API. HateBlocker then removes those images that are classified as hateful. 


<p align="center"> <img src="/img/pipeline.png"  width="500"> </p>


Here is also a short demo of the HB Chrome extention in a two cases with hateful and non-hateful memes. As you can see, the majority of hateful memes are filter out after the activation of the HB extention. 


[![demo](https://img.youtube.com/vi/ijJwfF7S91M/0.jpg)](https://www.youtube.com/watch?v=ijJwfF7S91M)


## [Multistream](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/MultiStream) model with [VisualBert](https://arxiv.org/abs/1908.03557) features
There have been several interesting developments in image captioning. I used a recently proposed model called [VisualBert](https://arxiv.org/abs/1908.03557) to extract features from image+caption and build a multi-stream classifier with the addition of features from Clarifai. This model gives AUC-ROC =0.75 of the validation data set.

## Some [ideas](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/MDVC) from [MDVC](https://arxiv.org/abs/2003.07758)
A very interesting development in video captioning is the recently published work by [Vladimir Iashin and Esa Rahtu](https://arxiv.org/abs/2003.07758). The idea of encoder-decoder architecture via a multiheaded attention connection seems very appealing to the problem of textual and visual feature fusion. I implemented a modified version of MDVC. The best performance with this model with only image and text features is AUC-ROC=0.66. Indeed, better performance can be achieved by incorporating the full set of features to form a stack of encoder-decoder networks [This is still in progress].
