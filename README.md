# Predicting Customer Churn with Neural Networks
This project demonstrates the application of neural networks for classifying audiobook customers into two categories: churned (0) or 'purchased again' (1), using TensorFlow and Keras. The workflow includes data preprocessing steps such as undersampling to handle class imbalance, scaling features with StandardScaler, and converting the dataset into TensorFlow tensors. A baseline neural network model is created and evaluated using a comprehensive training process with 30 runs, resulting in a test accuracy of around 80%. The project also explores hyperparameter tuning, testing various model configurations to improve performance. The results show robust performance with slight improvements in consistency and a reduction in standard deviation, demonstrating the model’s ability to generalize well across different settings.

One of the main goals of this project is to compare my approach with the instructor's code for the same problem. For this reason, I included a second Jupyter notebook, named "instructor's_approach". Ultimately, I believe my hands-on approach surpasses the instructor's in terms of code readability and comprehension. Additionally, I have automated the process more efficiently by passing almost all model parameters (except for the batch size) into a single function. My approach also allows for easy adjustment of the train, validation, and test split by changing only one parameter: the test percentage. Furthermore, I feed the model with 3 batched and prefetched datasets instead of 6, which enhances both performance and comprehension.

Regarding the numeric results, when using exactly the same parameters as the instructor, my approach generated very similar results, if not better. Specifically, my approach demonstrates better consistency, with a significant reduction in test accuracy and test loss standard deviations, and a slightly better test loss. However, it does produce slightly worse test accuracy.

*****************************************************************************************************************************************************************************************************************************************************
NOTE: I greatly acknowledge the Udemy course 'The Data Science Course: Complete Data Science Bootcamp 2024' (URL: https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/?couponCode=ST21MT121624) for sharing the dataset. If there are any restrictions regarding sharing this dataset through my GitHub, please contact me via Udemy at userID 'Kimon Ioannis Lappas' and I will immediately delete the dataset.
*****************************************************************************************************************************************************************************************************************************************************

## FILES INCLUDED:
1. The original Excel datafile that I used.
3. A '.py' file with the Python scipts I used for developing my model.
4. The core jupyter file.
5. The README file

   --> 4 Files in Total <--

## HOW TO SET UP THE ENVIRONMENT:
1. Download the project as a zip file or clone the repository to your local machine.
2. Open Anaconda Prompt and type:
   --> conda create -n 'YourEnvName' python=3.12.4 -c conda-forge numpy=1.26.4 pandas=2.2.2 scikit-learn=1.5.1 matplotlib=3.9.1 jupyterlab=4.2.4
3. Activate the newly-created environment.
4. Install additional packages from pip:
   --> pip install tensorflow==2.17.0 imbalanced-learn==0.12.3
5. Launch Jupyter via Anaconda Prompt.
6. Open the core jupyter file and run it.
7. Enjoy!

--> Thanks for your time! Feel free to connect with me on LinkedIn: linkedin.com/in/kimon-ioannis-lappas!!! <-
