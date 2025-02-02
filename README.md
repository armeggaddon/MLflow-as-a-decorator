# MLflow as a Decorator(MaaD)

*This repository demonstrates how to use MLflow monitoring tool to capture your ML pipeline activities without interrupting your actual workflow.*

### 1. What is [MLflow](https://mlflow.org/docs/latest/index.html)?

As per their documentation, "*MLflow, at its core, provides a suite of tools aimed at simplifying the ML workflow. It is tailored to assist ML practitioners throughout the various stages of ML development and deployment*". 

It is an open-source platform for managing the end-to-end machine learning lifecycle. It has several components that can be used to improve and streamline the machine learning pipeline. Here are some of the key uses of MLflow in an ML pipeline:

**Experiment Tracking**: MLflow allows data scientists to track experiments to record and compare parameters and results. This is essential for model development as it provides a systematic way to log all the experiments and their outcomes, which is critical for model selection and iteration.

**Project Packaging**: MLflow Projects component provides a standard format for packaging reusable data science code. It supports a variety of tools and frameworks and allows for the easy sharing of code between data scientists, and the execution of projects on different platforms.

**Model Management**: The MLflow Models component provides a convention for packaging machine learning models in multiple formats, enabling various deployment tools to use the model. This includes a simple REST API for model deployment.

**Model Serving**: MLflow allows for easy model deployment for serving predictions. This can be done either as a local REST API endpoint or on a cloud platform. MLflow integrates with tools like Microsoft Azure ML, Amazon SageMaker, and others.

**Model Registry**: A central model store, MLflow Model Registry manages the full lifecycle of an MLflow Model. It provides model lineage (which includes model versioning), model stage transitions (such as staging to production), and annotations.

**Collaboration**: MLflow's centralized model registry and experiment tracking facilitate collaboration among team members. Team members can view each other's experiments, share models, and transition models through various stages of development together.

**Reproducibility**: By tracking experiments, packaging projects, and standardizing models, MLflow helps in maintaining reproducibility in machine learning workflows. This is crucial when models need to be rebuilt or when results need to be verified.

**Integration with Existing Tools**: MLflow is designed to work with existing ML libraries, languages, and infrastructure. This means it can easily fit into existing workflows without requiring significant changes to the code or infrastructure.

**Scalability**: MLflow is built to scale from a single user running on a local laptop to hundreds of users collaborating across a multi-node production environment.

In summary, MLflow helps in managing the machine learning lifecycle, including experimentation, reproducibility, and deployment, which are critical aspects of building effective machine learning pipelines. It provides a unified platform to manage different stages of the ML pipeline and facilitates better collaboration and management of machine learning projects.

![alt text]([Isolated.png](https://github.com/armeggaddon/MLflow-as-a-decorator/blob/main/images/Experiemnt.JPG) "MLflow Experiment and runs")

### 2. What are the takeaways here?

Here I'm going to demonstrate a simple ML pipeline(not using Sklearn pipeline but just using few components of them in sequence) and showcase how we can leverage MLflow tool to capture the parameters, artifacts, metrics etc., of each stage.

Below are the list of components that are used as a workflow for demonstration

#### a. StandardScaler

It is a preprocessing tool in machine learning that standardizes features by removing the mean and scaling to unit variance. This scaler assumes that data follows a Gaussian distribution and scales them so they have a mean of zero and a standard deviation of one. 

Standardization can improve the performance of models, especially those sensitive to feature scaling, like support vector machines and neural networks. By making the features look like standard normally distributed data, the StandardScaler aids in the convergence of different algorithms and helps ensure that each feature contributes equally to the final model.

see more about [Standard Scaler](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html)

#### b. Principal Component Analysis (PCA)

 PCA is a statistical technique used to reduce the dimensionality of a dataset while retaining most of the variance. It transforms the data into a new set of orthogonal features called principal components, with the first component accounting for the largest possible variance, and each subsequent component having the next highest variance possible under the constraint that it is orthogonal to the preceding components. 
 
 PCA is useful for data visualization, noise reduction, and feature extraction. By reducing the number of features, PCA can simplify models and alleviate issues stemming from the curse of dimensionality.
 
 see more about [PCA](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html)
 
#### c. Logistic Regression

It is a statistical method used for binary classification. It models the probability that a given input belongs to a certain class (e.g., spam or not spam) by fitting data to a logistic function. The output is a value between 0 and 1, which is interpreted as the likelihood of the input being in the positive class. Parameters are estimated through maximum likelihood estimation, often using optimization algorithms like gradient descent. 

Logistic Regression is easy to implement, interpret, and can be extended to multiclass classification (Multinomial Logistic Regression) making it a staple in many machine learning applications.

see more about [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### 3. Lets get started...

The environment that we are working on should have the necessary libraries to run the code, so update the environment with the libraries present in *requirements.txt*

```console
pip install -r requirements.txt
```
Note: You can change the libraries to any latest version which is compatible for the code to work in your environment.

#### a. DataSets

I'm using sklearn iris dataset by importing them into the class MlFlowDemo during initialization. Along with that I'm splitting the data sets into training and testing through train_test_split.

Below code showcases the initialization of class.

```python
class MlFlowDemo():
    
    def __init__(self, exp_name, **kwargs):
        
        self.classifier = LogisticRegression()
        self.exp_name = exp_name
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42
```		

#### b. Methods
Now I'm creating different methods for preprocessing, feature selection and classification under the same class.
 
In any ML pipeline all the steps except the last one are called as transformers since they not only fit the data, they perform transformation as well for the next step to progress. Whereas the last step in the workflow is called as estimator, since it generally estimate the final output, hence they accompany with the predict method in it.

In preprocessing, I'm using StandardScaler() and apply the fit_transform method during training, whereas in prediction we only need to transform the given data by using the existing scaler instance. The same logic is applicable to PCA as well,

```python
    def preprocessing(self, X, y, train=True, **kwargs):
        if train:
            self.preprocessor = StandardScaler()
            self.operator = self.preprocessor
            xt = self.preprocessor.fit_transform(X,y)
            return xt
        else:
            self.operator = self.preprocessor
            xt = self.preprocessor.transform(X)
            return xt
```

In case of Logistic Regression, we generate the model instance during training and getting the accuracy score in prediction using the generated model instance.

```python
    def classification(self, X, y, predict=False, **kwargs):
        if not predict:
            self.operator = self.classifier
            self.model = self.classifier.fit(X, y)
        else:
            self.operator = self.classifier
            y_pred = self.model.predict(X)
            self.accuracy_score = accuracy_score(ml_instance.y_test, y_pred)
            return y_pred
```

#### c. Add MLflow as a Decorator

We have established the basic workflow, and now we need to integrate MLflow to capture workflow details. To start, we should create an experiment to log the flow within a specific entity. Generally, many examples demonstrate the use of MLflow via its direct instance. However, that approach does not allow for multiple active instances, leading to issues with parallel processing. To resolve that, this demo illustrates how to use the MLflowClient directly to log ML activities.

Additionally, incorporating MLflow components directly into the code can make it difficult to separate the business logic from the ML monitoring layer. If you want to replace one ML monitoring layer with another in the future, you would have to modify the entire codebase, which is cumbersome and time-consuming.

To address these challenges, I recommend keeping the ML monitoring layer separate from the business logic by using decorators. In Python, a decorator is a design pattern that enables you to add new functionality to an existing object without altering its structure. Decorators are usually implemented as functions (or classes) that take another function (or method) as an argument, extend its behavior, and then return a new function with the added functionality.

#### d. ML flow

It is nearly impossible to present the entire code implementation in this documentation, so I will provide only the highlights here. Please refer to the source code for more detailed information.

As mentioned earlier, we need to start by creating an experiment. Following that, we create a parent run, within which all the child runs, such as preprocessing, will occur. We then capture each child run, tagging it with the parent run to ensure that the child run is correctly linked to the parent run in the UI.

During this process, we capture the necessary artifacts, parameters, metrics, and the model. The code is structured with these elements in each decorator, and to activate it, we add these decorators as annotations to the respective methods in the pipeline code. This code will be executed whenever the actual workflow is triggered, logging the data according to the decorator logic. Additionally, we can disable ML logging by setting a boolean parameter.


#### e. Invoke the workflow

Now we have everything in place, we will start the workflow by instantiating the class and the call all the methods in sequence. First by calling the workflow using the train data that we extracted using train_test_split which will create instances of Scaler, PCA and a classification model.

Once the training workflow is done we call the prediction flow by passing the test data to the trained instances. In the real case scenario, the trained instances will be stored as a model pipeline and hosted as a service which will be used for prediction.

##### Note : *In this demo, we are not hosting the trained model as an instance, since that requires a database where the model will be saved either as an inbuild model type like sklearn, pytorch, tensorflow or a generic pyfunc model, To know more about that feel free to contact me or drop me a message.*

Run the workflow multiple times by changing the parameters. This will generate a folder call mlruns if MLflow is enabled. To view the ML logging in UI run the below command from the mlruns directory and compare the accuracy between different runs.

```console
    mlflow ui
```
