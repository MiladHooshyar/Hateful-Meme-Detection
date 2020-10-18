

# HateBlocker

<p align="left"> <img src="/img/logo.png"  width="70"> </p>


HateBlocker is an API that utilizes deep learning to detect hateful memes from visual and textual information.


## Table of Contents
  * [Problem description](#problem-description)
  * [Data](#data)
  * [Models](#models)
    + [Baseline](#baseline)
    + [Feature fusion and dense classifier: Concat model](#feature-fusion-and-dense-classifier:-concat-model)
    + [Multistream model with VisualBert features](#multistream-model-with-visualbert-features)
    + [Some ideas from MDVC](#some-ideas-from-mdvc)

# Problem description
The effect of hateful content on social media can be devastating. Results of a [recent poll](https://www.huffpost.com/entry/social-media-harassment-fake-news-poll-alex-jones_n_5b7b1c53e4b0a5b1febdf30a) have shown that the majority of US adults consider hateful content a serious problem for which social media companies bear the responsibility of detecting and removing. In this project, I built a hateful meme detector (HateBlocker) and deployed the model as a stand-alone API and a Chrome extension.

# Data
The data for this project come from [Facebook research](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/), which consists of 8.5K training and 0.5 validation examples. This data set consists of image and caption, which were run through couple of pre-trained features extraction models (see below for details). 

## Setup
To use this model, first clone the repo using
```
git clone https://github.com/MiladHooshyar/Hateful-Meme-Detection.git
```

Use pip to install the required [libraries](https://github.com/MiladHooshyar/Hateful-Meme-Detection/blob/master/requirements.txt)

Download the features from [here](https://drive.google.com/file/d/1ikgWVV45L7rsgQ6y80721VyyzWnE3fRo/view?usp=sharing) and place them in the [/data](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/data) folder. For more detalis about how to run specific models, see below.



# Models
This project has been carried out in collaboration with the research team at [Clarifai](https://www.clarifai.com/). Thus the focus was to use the pre-trained models of the Clarifai platform to extract the image and text features. In addition, I performed experiments inspired by recent developments in image and video captioning (e.g. [VisualBert](https://arxiv.org/abs/1908.03557) and [MDVC](https://arxiv.org/abs/2003.07758))

## [Baseline](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/BaseLine)
The baseline model is a simple logistic regression that is trained on top of the image and caption features from the Clarifai platform. To run the baseline model, try

```
python /Baseline/LR_train.py
```
This model yeilds AUC-ROC=0.63 on validation set.

## [Feature fusion and dense classifier: Concat model](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/Concat)
This classifier is trained on top of several feature sets, including image, text, image moderation, and text moderation features. The fifth stream of information is supplied by concatenating the top-five concepts of the image to the image caption and then feeding that to the text feature extraction network.

<p align="center"> <img src="/img/model.png"  width="500"> </p>

To run the Concat model, try

```
python /Concat/Concat_train.py
```
The best performance of this model is AUC-ROC = 0.71 on validation set. Here is an example that shows how the intension of a meme changes with modifying either the textual and visual content, which highlights the nonlinear interaction of information from image and text. The probibility of hatefulness from Concat model is denoted by $P-H$. 


<p align="center"> <img src="/img/example1.png"  width="500"> </p>


## [Multistream](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/MultiStream) model with [VisualBert](https://arxiv.org/abs/1908.03557) features
There have been several interesting developments in image captioning recently. I used the newly proposed model [VisualBert](https://arxiv.org/abs/1908.03557) to extract features from image+caption and build a multi-stream classifier with the addition of features from Clarifai. 

To run this model, try

```
python /Multistream/Multistream_train.py
```

This model gives AUC-ROC =0.75 of the validation data set.

## [Ideas](https://github.com/MiladHooshyar/Hateful-Meme-Detection/tree/master/MDVC) from [MDVC](https://arxiv.org/abs/2003.07758)
A very interesting development in video captioning is the recently published work by [Vladimir Iashin and Esa Rahtu](https://arxiv.org/abs/2003.07758). The idea of encoder-decoder architecture via a multiheaded attention connection seems very appealing to the problem of textual and visual feature fusion. I implemented a modified version of MDVC. To run this model, try

```
python /MDVC/MDVC_train.py
```

The best performance with this model with only image and text features is AUC-ROC=0.66 on the validation data. Indeed, better performance can be achieved by incorporating the full set of features to form a stack of encoder-decoder networks [This is still in progress].


## Deployment
The Concat model is coupled to the google vision OCR (for caption extraction) and is implemented as a Flask API. The API inputs a meme URL and outputs the class probabilities. The end-product of this project is a Chrome extension (HateBlocker), which extracts meme URLs and sends them through the API. HateBlocker then removes those images that are classified as hateful. 


<p align="center"> <img src="/img/pipeline.png"  width="500"> </p>


Here is also a short demo of the HB Chrome extension in two cases with hateful and non-hateful memes. As you can see, the majority of hateful memes are filtered out after the activation of the HB extension. 


[![demo](https://img.youtube.com/vi/ijJwfF7S91M/0.jpg)](https://www.youtube.com/watch?v=ijJwfF7S91M)


