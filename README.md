_**Image Classification with CNNs on iNaturalist Dataset:**_

This project focuses on using Convolutional Neural Networks (CNNs) to classify images in a subset of the iNaturalist dataset. The dataset contains a large collection of images of various plant and animal species.

_**Part A: Building a 5-Layered CNN from Scratch**_

In the first part of the project, a 5-layered CNN was built from scratch to classify images in the iNaturalist dataset. The trained model was evaluated on a test set to measure its accuracy in correctly classifying the images.

_**Part B: Transfer Learning using a Pre-Trained Model**_

In the second part of the project, a pre-trained model (ResNet50) was used for the same classification task. The aim of this part of the project was to demonstrate the concept of transfer learning, where a pre-trained model is fine-tuned to perform a specific task. The ResNet50 model was fine-tuned on the provided subset of the iNaturalist dataset and its performance was evaluated on the test set.

The results showed that the fine-tuned model was able to provide double the accuracy of the model trained from scratch, demonstrating the power of transfer learning.

_**Tools Used:**_

The code for this project was developed using Google Colab notebooks, which provided a GPU for faster training. The following Colab files were used for development:

cs6910_assignment_2_partA.ipynb - Contains code for Part A of the assignment

cs6910_assignment_2_partB.ipynb - Contains code for Part B of the assignment

Once the preliminary version of the code was developed in the Colab notebooks, it was transferred to the following Python files for training purposes:

cs6910_assignment_2_PartA.py

cs6910_assignment_2_PartB.py

_**Evaluation Metrics:**_

The performance of the models was evaluated using accuracy as the metric. The accuracy was calculated as the ratio of correctly classified images to the total number of images in the test set.

_**Conclusion:**_

This project demonstrated the effectiveness of CNNs for image classification tasks and the power of transfer learning using pre-trained models. The results show that fine-tuning a pre-trained model can significantly improve the prediction accuracy in classifying images.



