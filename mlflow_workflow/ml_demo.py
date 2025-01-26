#REQUIRED IMPORTS FOR THE WORKFLOW TO WORK
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow_workflow.ml_utilities import mlflow_tracking, mlflow_stage

#CLASS FOR PERFORMING THE OPERATION
class MlFlowDemo():
    
    #INITIALIZING THE CLASS WITH TRAIN AND TEST DATA FROM SKLEARN DATASETS
    @mlflow_tracking
    def __init__(self, exp_name, **kwargs):
        
        self.classifier = LogisticRegression()
        self.exp_name = exp_name
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #STANDARD SCALER FROM SKLEARN ACTING AS A PREPROCESSOR
    @mlflow_stage
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
     
    #PRINCIPLE COMPONENT ANALYSIS ACTS AS FEATURE SELECTOR FOR THE WORKFLOW   
    @mlflow_stage
    def feature_selection(self, X, y, train=True, **kwargs):
        if train:
            self.feature_sel = PCA(n_components=2)
            self.operator = self.feature_sel
            xt = self.feature_sel.fit_transform(X, y)
            return xt
        else:
            self.operator = self.feature_sel
            xt = self.feature_sel.transform(X)
            return xt
    
    #LOGISTIC REGRESSION WHICH FROM SKLEARN ACTS AS A CLASSIFIER
    @mlflow_stage
    def classification(self, X, y, predict=False, **kwargs):
        if not predict:
            self.operator = self.classifier
            self.model = self.classifier.fit(X, y)
        else:
            self.operator = self.classifier
            y_pred = self.model.predict(X)
            self.accuracy_score = accuracy_score(ml_instance.y_test, y_pred)
            return y_pred
            
##################
#TRAINING WORKFLOW
##################
#INSTANTIATING THE DEMO INSTANCE
enable_mlflow=True
preprocess_fn = "preprocess.txt"
pca_fn = "pca.txt"
classified_fn = "classify.txt" 

ml_instance = MlFlowDemo("MlflowDemo", mlflow_flag=enable_mlflow)

#INVOKING PREPROCESSOR
preprocessed_data = ml_instance.preprocessing(ml_instance.X_train, ml_instance.y_train, stage_name="train_preprocess", file_name=preprocess_fn)
#IDENTIFYING THE FEATURES
selected_features = ml_instance.feature_selection(preprocessed_data, ml_instance.y_train, stage_name="train_pca",file_name = pca_fn)
#BUILDING THE CLASSIFICATION MODEL
ml_instance.classification(selected_features, ml_instance.y_train, stage_name='train_classify')

####################
#PREDICTION WORKFLOW
####################
#PREPROCESSING THE TEST DATA
preprocessed_data = ml_instance.preprocessing(ml_instance.X_test, y=None, train=False, stage_name="pred_preprocess", file_name = preprocess_fn)
#FEATURE SELECTION
selected_features = ml_instance.feature_selection(preprocessed_data, y=None, train=False,stage_name="pred_pca", file_name = pca_fn)
#MODEL PREDICTION
y_pred = ml_instance.classification(selected_features, y=None, predict = True, stage_name="pred_classify", file_name = classified_fn)

#COMPUTING ACCURACY

accuracy = accuracy_score(ml_instance.y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
