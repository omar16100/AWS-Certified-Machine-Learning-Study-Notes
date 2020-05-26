# AWS-Certified-Machine-Learning-Study-Notes

AWS Certified Machine Learning – Study Notes

> These notes are written by a data scientist, so some basic topics may be glanced over

## 1. [Machine learning concepts](01-concepts.md)

* Machine learning lifecycle
* Supervised vs Unsupervised vs Reinforcement learning
* Optimisation
* Regularisation (L1 Lasso & L2 Ridge)
* Hyperparameters
* Cross-validation

## Machine learning life-cycle

![life cycle](img/1-life-cycle.png)

## Supervised, unsupervised & reinforcement learning

## Optimization

Gradient decent

## Regularization

> Need to understand, not calculate
Mathematical process which will try and desensitize your model to a particular dimension.

* L1 -- Lasso
* L2 -- Ridge regression
* Use when model is overfit

## Hyperparameters

External parameters set BEFORE model is trained, such as learning rate, epochs and batch size

* **Learning rate**: Size of step taken in gradient decent (between 0 and 1)
* **Batch Size**: Number of samples used to train. One, some or all of your data (commonly 32/64/128). Could be based on infrastructure.
* **Epochs**: Number of times your algorithm will process all the training data. Contains one or more batches. Very high numbers, 10-1000

Hyperparameter tuning strategies

* random search
* bayesian search

## Cross-validation

TODO

## 2. [Data](02-data.md)

* Feature selection
* Feature engineering
* Principal Component analysis (PCA)
* Missing and unbalanced data
* Label encoding & One-hot-encoding
* Train-test splits & Randomisation
* RecordIO format

## Feature selection and engineering

**Selection**:

* Choose features that don't impact the model performance
  * person's name when predicting if they like tea
* Makes model faster to train and more accurate
* Gaps in data. Remove? Keep? Infer?

> Use domain knowledge, drop features with little correlation target, low variance or lots of missing data

Feature engineering

* Compute new features from existing features
  * Time of day, from timestamp
  * Country from city

> Simplify features, remove irrelevant info, standardise ranges (0-10: 0-1 && -100-100: -1-1), transform data

## Principal component analysis

Unsupervised dimension reduction while retaining all or most of the information

* Example is taking a photograph of a 3d object. You lose one dimension of the data, but the info is still there
* Often used as a data pre-processing step
* Can be used to plot high-dimensional data as groups of features

## Missing and unbalanced data

Missing:

* Few datapoints missing: Replace missing data with average
* Few rows missing: Remove row
* Column missing most data: Remove column

How to deal with Unbalanced:

* Get more data (often overlooked)
* Oversample minority: Little variation
* Synthesize data: Take minority data, apply some variation to make new points
* Different algorithms

## Label and 1-hot-encoding

* Label-Encoding: Replace strings with values (brazil:0, USA: 1, UK: 2)
* One-hot-encoding: New columns with binary values if they match (Brazil: (brazil:1, usa:0, uk: 0) )

## Splitting and Randomization

Train/test splits. ALways randomize data

## RecordIO format

Pipe mode instead of File mode. Faster start and better throughput. Streams the data to the model

* SageMaker works best with RecordIO, streams data directly from S3 without storing locally

## 3. [Machine learning models](03-machine-learning-models.md)

* Logistic Regression
* Linear regression
* SVM
* Decision trees
* Random Forest
* K-means
* KNN
* Latent Dirichlet ALlocation - LDA

TODO

## Latent Dirichlet ALlocation - LDA

> Not same as Linear Discriminant analysis (LDA)
Text analysis, can be used for topic analysis of text documents

## 4. [Deep Learning](04-deep-learning.md)

* Neural Networks
* Activations functions (sigmoid, Tanh, ReLU)
* Weights & biases
* Forward & Back propogation
* Convolutional Neural Networks (CNN)
* Filters
* Transfer Learning
* Recurrent Neural Networks (RNN)

## 5. [Model Performance and Optimization](05-model-performance.md)

* Sensitivity (Recall / TPR)
* Specificity (TNR)
* Precision
* Accuracy
* ROC / AUC
* F1 Score
* Gini impurity

Confusion matrix

![Confusion matrix](img/confusionmatrix.jpg)

[source](https://manisha-sirsat.blogspot.com/2019/04/confusion-matrix.html)

## Precision vs Recall

![Precicion vs Recall](img/Precisionrecall.svg)

[source](https://en.wikipedia.org/wiki/Precision_and_recall)

**Sensitivity** (Recall / True Positive Rate): Number of positives out of all positives

* TP / (TP + FN)
* When you want to avoid **false negatives**
* % of people with a disease that are identified as having the disease
* _"Recall all the **positives** in the dataset, how many did you get right?"_

**Specificity** (True Negative rate): Correct  positives out of the predicted positive results

* TN / (TN + FP)
* When you want to avoid **false positives**
* % of people without a disease that are identified as not having the disease
* _"Recall all the **negatives** in the dataset, how many did you get right?"_

**Precision**:

* TP / (TP + FP)
* How accurate are you?
* _Out of all the things you said were positive, how many actually were?_

**Accuracy**: Proportion of all predictions correctly identified

* (TP + TN) / Total
* Correct / total = Accuracy
* How correct overall am I?

**ROC / AUC**: Visualise the balance of sensitivity (TPR) and specificity (FPR) of a  binary classifier when using different cutoff points for the classification

<https://www.youtube.com/watch?v=4jRBRDbJemM>

| ![roc-max](img/5-roc-max-sens.png) | ![roc](img/5-roc-roc.png) |
| --- | --- |

**GINI Impurity**: Measure of impurity when evaluating which feature to use in splitting decision trees. The feature with the lowest GINI impurity score gets chosen as the root node.

**F1 Score**: Combination of Recall and precision, takes more into account FP and FN than accuracy.

* 2 *( Recall* Precision) / ( Recall + Precision )

## 6. [Machine Learning Tools and Frameworks](06-tools-frameworks.md)

* Pytorch & Scikit-learn
* Tensorflow & Keras
* MXNET & Gluon
* Tensors & Graphs

Machine learning & Deep learning
![ML-Framework](img/6-ml-framework.png)
![DL-Framework](img/6-dl-frameworks.png)

## Tensorflow

Define graph up front and then run it

* Tensor: Multidimensional array that can hold data
* Graph: Flow of data

```python
import tensorflow as tf
```

## Pytorch

Creating the graph as you go along. Can be more dynamic than TF.

```python
import pytorch
```

## MXNet

Framework of AWS Sagemaker, similar to pytorch you create graphs on the fly.

```python
import mxnet as mx
from mxnet import nd #ndarray
from mxnet import autograd #autogradient
```

## From Linux academy

### TensorFlow

TensorFlow (backed by Google) is an end-to-end, open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources. This lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

_(Source: <https://www.tensorflow.org/)>_

### Keras

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

_(Source: <https://keras.io/)>_

__Or, to put it another way:__

TensorFlow is a complex tool. Keras has been built on top of TensorFlow as a more user-friendly interface. It helps us rapidly prototype models, and we use it in this lab.

## 7. [AWS Services](07-aws-services.md)

* S3 Datalakes
* Kinesis (video stream / data stream / firehose / data analytics)
* Glue
* Athena
* Elastic Map Reduce (EMR) & Spark
* EC2 instance types for ML
* AWS Machine Learning service (deprecate)

## S3 datalakes

Collection of structured and unstructured files stored in S3
![Datalake](img/7-datalake.png)

## Kinesis sample architecture

**Kinesis** (or other) captures data > **S3** stores it > **Glue** catalogs it >  **Athena** (simple) and **EMR** (large/complex) queries it > **Sagemaker** models it
![Datalake 2](img/7-datalake-arch2.png)

* Kinesis Data Analytics can be used to process and analyze streaming data using SQL or Java. You can use Kinesis Data Analytics to create a leaderboard for an online game and you can calculate metrics in real time.

## Glue

Creates catalogues of data, doesn't store any data. Gluing together multiple schemas, databases, sources and load it into a final destination.

* Some limited behind the scenes machine learning capabilities to assist, such as detecting duplicate users (work/personal email etc)

Key components

* etl operations
* jobs system
* crawlers and classifiers
* data catalog

## Athena

"SQL Interface for S3".

* Can save outputs of queries back to S3

## Quicksight

AWS Business Intelligence (BI) tool. Should be enough to know from high level.

* Create dashboards, email reports and embedded reports
* Analyse your initial training data

## Kinesis - mentioned a lot

Ingest large amounts of data from different streams. IoT devices is a common use-case.

* **Video stream**: Can store video for up to 7 days and playback
* **Data stream**: Catch all for processing
* **Firehose**
  * Ingest real-time data stream it to s3, Redshift, es & splunk
  * Can transform data before delivering
* **Data Analytics**: Analyse data on the fly using SQL queries

![Kinesis Overview](img/7-kinesis-overview.png)

### Example

Video streaming and facial recognition on the fly that sends out notifications if a particular person is spotted in a crowd.  
![Kinesis Video](img/7-kinesis-video.png)

## EMR w/ SPARK

**Elastic Map Reduce**: Hosting Massively parallel compute, works very well in cloud. Integrates well with S3.

* Task nodes can be ripped out without failing the whole cluster == Spot instances

**Spark**: Fast analytics engine for massively parallel compute tasks.

* Can run inside EMR & Sagemaker

## EC2 for ML

**EC2 instances for ML**:

* Compute optimised
* Accelerated Computing (GPU)
* Sagemaker specific: _ml.*_

**Conda based deep learning AMIs**: TensorFlow, keras, MXNET, gluon, pytorch, ...

**Service limits** on by default to protect you from accidentally spinning up a large host of very expensive instance for terabytes of data.

* Can be lifted, but have to be requested (takes days)

## AWS Machine Learning (ML) - Service, deprecate

The MVP ML service, but no longer exists. Comes up as a red-herring, never the right answer

## 8. [AWS Application Services AI/ML](08-aws-applications-ai-ml.md)

* Rekognition (images)
* Rekognition (videos)
* Polly (text2speech)
* Transcribe (speech2text)
* Translate
* Comprehend
* Lex (chatbots)
* Step Functions

## Rekognition

Fully managed service, serving a pre-trained deep learning model for image and video analysis.

* Common to use AWS Lambda to connect to the services
* **Images**: image moderation (eg. porn filter), facial analysis (find faces, gender), celebrity, comparison, text in image
* **Videos**:  Detect people of interest, create metadata catalog for stock videos, detect offensive content

### Video architecture

* Object is uploaded to S3
* Triggers a lambda function to call the Rekognition service
* Rekognition queries S3 to get the video and starts analysing
* When done, will send a message through SNS topic that it's done
* SNS will put that through an SQS
* Another Lambda function triggered when the SNS topic is triggered, get metadata from SQS about the video results, and finally go to Rekognition and get the results

![video](img/8-rek-video-arch.png)
![video](img/8-rek-video-arch-stream.png)

## Amazon Polly

Text to speech service (TTS). Fully managed pre-trained deep learning. Supports multiple languages, male or female voices and custom lexicons for industry specific language.  

* **Speech Synthesis Markup Language** (SSML): Be more specific about the inflections (eg. whispering)
* **Use cases**: Read out loud web content, provide generated announcements, automated voice response solutions

## Amazon Transcribe

Speech to text. Automatic Speech Recognition (ASR). Pre-trained deep learning models, supports different languages and custom vocabulary

## Amazon Translate

Text translation, fully managed pre-trained deep learning models. Batch process, real-time, custom terminology.

* **Use cases**: Enhance online customer chat application to translate conversations in real time, batch translate documents, create news publish solutions to multiple languages
* Knows more languages than other services, such as Comprehend

## Amazon Comprehend

Natural language Processing (NLP) system. Give it text, get back analysis of the text.

* **Key Offerings**: Key-phrase extraction, sentiment, syntax, entity recognition, language detection, topic modelling, multiple languages.
* Medical named entity and relationship extraction
* Customise: entities and classification
* Use case: Customer sentiment analysis, label unstructured data, topics from transcribed audio

## Amazon Lex

Chatbots, Alexa.

## Service chaining w/ AWS Step Functions

Orchestrate multiple Lambda functions

### Example architecture with Step functions

* Audio is saved into an **S3 Bucket**
* **Lambda-1** is triggered, that simply calls the **Step Function** (red box)
* Inside the **Step Function**, is a **Lambda** that calls both **Transcribe** & a **Wait function** (clock icon)
* **Transcribe** then calls **S3** to get back the audio file and starts processing it. This might take a while
* The middle **Lambda-2** has 1 role, to query **Transcribe** for the job status (eg. every 10seconds) and pass the status and data into a **Decision Point** (question mark)
* If the status isn't finished the **Decision Point** invokes the **Wait Function** again, which restarts the check status process
* If the status is finished, invoke the last **Lambda-3** that invokes **Comprehend**
* **Comprehend** analyses the data, sends the results back to **Lambda-3** that passes it on to another service

![Step Architecture](img/8-step-architecture.png)

## 9. Sagemaker -- VERY IMPORTANT TOPIC

## [Intro](09a-sage-intro.md)

* Sagemaker High Level
* Three stages: Build, train, deploy
* Sagemaker console
* Sagemaker API
* Sagemaker Python SDK

Note really like any other service. Could spend all your time in there without having to do anything else. Fully managed service from start to finish. End-to-end

![sagemaker definitions](img/9-sagemaker-definition.png)

## Three stages

**Build**: Preprocess, ground truth, notebooks

**Train**: Built-in algorithms, hyperparameter tuning, notebooks, infrastructure

**Deploy**: Realtime, batch, notebooks, infrastructure, Neo

## Control

Console, SDK, Jupyter

**Sagemaker API**: Can be called to provision and run services

**Sagemaker Python SDK**: Control and provision Sagemaker instances right through Jupyter Notebooks

![Sage sdk](img/9-sage-sdk.png)

## Sagemaker notebooks

Instance type doesn't have to be very large in order for you to do intense machine learning.

* Instance type can be used as a control panel for the notebook, and the notebook calls services that provision more intense infrastructure to run the code
* Lots of example notebooks to browse
* Starting notebooks takes a while as it's provisioning all the infrastructure in the backend
* Clicking `Open Jupyter | Open Jupyterlab` redirects you to the endpoint for your instance with a pre-signed url.
  * Can't share the link with others

## [Sagemaker Build](09b-sage-build.md)

* **!!Define your problem first!!**
* **Build process**: Visualise, Explore, Feature engineering, Synthesize data, Convert data, Change structure (joins), Split data
* Ground truth
* **SageMaker Algorithms**: Built in, marketplace, custom
* **Algorithm Types**: eg. BlazingText (AWS-Comprehend), Image classification (AWS-Rekognition)

Have a goal in mind before starting any machine learning task. **Define your problem first**

## Data Preprocessing

Can take as long as you want, making it very important that we define our problem upfront. Primarily use SageMaker notebooks

* Visualise
* Explore
* Feature engineering: Can be done in Jupyter if its small, else EMR
* Synthesize data
* Convert data
* Change structure (join)
* Split data

## Ground truth

> _"Build highly accurate training data using machine learning and reduce data labeling costs by upt to 70%"_

Uses active learning to help label the most ambiguous data and outsources that to Mechanical Turk

![Ground Truth](img/9b-ground-truth.png)

## SageMaker Algorithms

* **AWS built-in** algorithms
* **AWS Marketplace**: Crowd sourced, sometimes pre-built for transfer learning
* **Custom**: Make your own

Algorithm Types

* **BlazingText**: Word2vec (Amazon Comprehend)
* **Image classification**: CNN (Amazon Rekognition)
* **K-means**: Based off web-scale k-means clustering
* LDA
* PCA
* XGBoost

## [Sagemaker Train](09c-sage-train.md)

* Architecture behind Sagemaker training: Algorithms stored in docker containers in ECS, spin up EC2 instances
* **AWS Marketplace**: **Algorithms** are to be trained, **Model packages** are pre-trained
* **Where to access data**: S3, EFS, FSx for Lustre
* **Filetypes**: Files / Pipe (recordIO)
* **Instance types**: ml.m4, ml.c4, ml.p2 (gpu)
* Some algorithms only support GPU instances
* Managed spot training & Checkpoints
* Automated Hyperparameter tuning

## Sagemaker architecture for training

When you start a sagemaker training job AWS calls ECS and the containers that contain the algorithms. We also need data, generally stored in S3.

* We define the algorithms and where the data lives
* Sagemaker then spins up the EC2 instances it needs, pulls in the data, trains the model using the models available in ECS.

**AWS Marketplace**: **Algorithms** are to be trained, **Model packages** are pre-trained

**Where to access data**: S3, EFS, FSx for Lustre (High performance, high throughput for High Performance Computing -- HPC)

**Filetypes**: Files / Pipe (recordIO)

Instance types

* Some algorithms only support GPU instances (ml.p2 family)
* AWS recommends: ml.m4, ml.c4, ml.p2 (GPU)
* GPUs more expensive, but faster

**Managed spot training**: Optimise cost of training models up to 90% over on-demand using spot instances, using **Sagemaker Checkpoints** in the model state

## Hyperparameter tuning

Choose model, set ranges of hyperparameters (max_depth:3-9), choose performance metric to measure (maximise AUC)

* There's a machine learning model, that monitors which hyperparameters are working the best and tunes the models accordingly.

## [Sagemaker Deploy](09d-sage-deploy.md)

* Real-time inference
* Batch inference

**Inference pipelines**: Unseen data goes into model which makes a prediction

* But often we have models feeing into models
* Unseen data comes in, is pre-processed, the fed into PCA, output of PCA goes into XGBoost, which makes the final prediction  

![Inference Pipeline](img/9d-inf-pipe.png)

## Real-time  inference

* Model gets data from S3, model built in ECR and accessed via SageMaker endpoint
* SageMaker endpoint. Not accessible outside of AWS, only for internal usage. Have to be authenticated. Called via API

![Inference Realtime](img/9d-inf-real.png)

## Batch inference

* Model gets data from S3, model built in ECR and create a Batch Transform job
* Batch job takes a large volume of data, pushes it into the model, makes inferences and stores the output somewhere

![Inference Batch](img/9d-inf-batch.png)

## 10. [Security](10-security.md)

* Sagemaker root access
* **AmazoneSageMakerFullAccess** policy: Admin access to SageMaker + necessary access to other services
* Sagemaker can see objects in S3 by default, can't access
* Deployed into **public VPC** by default

* **IAM policy**: sagemaker:CreatePresignedNotebookInstanceUrl
* Have to option to grant Root access to the machine that's running the notebook instance.
  * Might be a security risk.
  * Can restrict the access if needed
  * Can associate a lifecycle script that runs when the notebook is started
  * Doesn't support resource based policies

* **AmazoneSageMakerFullAccess** policy: Grants admin access to SageMaker + specific access to a lot of other services such as cloud watch and logs
  * Eg. can SEE objects in S3, but can't access them

## Sagemaker and VPC

Sagemaker deploys to a public VPC by defaults

* Useful to get packages
* For security sake we might want to create a custom VPC with a private subnet and deploy it into that. We can then configure it with S3 VPC endpoint or NAT Gateway.

## Other

* **AWS DeepLens** – Deep learning enabled video camera for developers
* **AWS DeepRacer** - Reinforcement learning enabled race-car

## Sagemaker FAQs notes

* **CloudTrail** to see SageMaker API calls
* **Notebooks persist** on the volume of the attached instance. So stopping the instance doesn't make you lose your progress.
* **Managed spot training** uses Spot instance to train. Have to specify time to wait for spot capacity
  * Good when you have flexibility
  * Uses checkpoints to store progress. Avoids failure when instance is terminated.
* BlazingText
* **Automated hyperparameter tuning** available for all algorithms (including custom one).
  * Uses a custom Bayesian Optimization under the hood
* Can currently only optimise for one objective (ie. accuracy or speed)
* **Reinforcement learning** is a machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences
  * Available to train in SageMaker. Can use **AWS RoboMaker**, Open AI Gym or commercial simulation environments to train
* **SageMaker Neo**: Enables machine learning models to train once and run anywhere in the cloud and at the edge
  * Optimizes models built with popular deep learning frameworks that can be used to deploy on multiple hardware platforms
  * Two major components – a **compiler** and a **runtime**
  * Supports the most popular deep learning models for computer vision and decision tree models:
    * AlexNet, ResNet, VGG, Inception, MobileNet, SqueezeNet, and DenseNet models trained in MXNet and TensorFlow
    * classification and random cut forest models trained in XGBoost
* Model performance from multiple runs is available in the Management Console in tabular form giving you a **leaderboard**
* Can't directly access the underlying hardware SageMaker runs on
* Can scale manually, or automatically using **Application Auto Scaling**
* **CloudWatch Metrics** to monitor SageMaker environment
  * Logs written to CloudWatch

## SageMaker Algorithms - Overview

* **Built-in algorithms**:
  * linear regression
  * logistic regression
  * k-means clustering
  * principal component analysis (PCA)
  * [factorization machines](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html)
    * A factorization machine is a general-purpose supervised learning algorithm that you can use for both classification and regression tasks. It is an extension of a linear model that is designed to capture interactions between features within high dimensional sparse datasets economically. For example, in a click prediction system, the factorization machine model can capture click rate patterns observed when ads from a certain ad-category are placed on pages from a certain page-category. Factorization machines are a good choice for tasks dealing with high dimensional sparse datasets, such as click prediction and item recommendation.
  * neural topic modeling
  * latent dirichlet allocation
  * gradient boosted trees
  * sequence2sequence
  * amazon forecast : is a fully managed service for time series forecasting (retail, financial planning, supply chain, healthcare, inventory management).
  * word2vec
  * random cut forest : an unsupervised algorithm for detecting anomalous data points within a data set. These are observations that diverge from otherwise well-structured or patterned data. Anomalies can manifest as unexpected spikes in time series data, breaks in periodicity, or unclassifiable data points.
  * image classification
* **Optimized containers**:
  * Apache MXNet
  * Tensorflow
  * Chainer
  * PyTorch
* **Custom algorithms** by using Docker images

## References

1. [Linux Academy](https://linuxacademy.com/cp/modules/view/id/340)
1. [SageMaker FAQ](https://aws.amazon.com/sagemaker/faqs/)
1. Blog Posts
    * [Passing the AWS Certified Machine Learning Specialty Exam](https://blog.thecloudtutor.com/2019/03/18/Passing-the-AWS-Certified-Machine-Learning-Specialty-Exam-MLS-C01.html)
