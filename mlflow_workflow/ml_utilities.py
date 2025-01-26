import os
import time
import pickle
import inspect
import tempfile
from sklearn import metrics
from mlflow.entities import Param, Metric
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.tracking.context.default_context import DefaultRunContext
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID



def mlflow_tracking(func):
    
    def create_mlflow(self, *args, **kwargs):
        
        mlflow_flag = kwargs.get('mlflow_flag',False)
        self.mlflow_flag = mlflow_flag
        
        if self.mlflow_flag:
        
            try:
                exp_name = args[0]
                mlclient = MlflowClient()
                try:
                    exp_id = mlclient.create_experiment(exp_name)
                except MlflowException as e:
                    print(e)
                    exp_id = mlclient.get_experiment_by_name(exp_name).experiment_id
                  
                run_tag = DefaultRunContext().tags()  
                run_tag.update({MLFLOW_RUN_NAME:"pipeline"})
                parent_run = mlclient.create_run(experiment_id=exp_id,tags=run_tag)
                parent_run = parent_run.info.run_id
                self.parent_run = parent_run
                self.exp_id = exp_id
                
                out = func(self, *args, **kwargs)
                mlclient.set_terminated(parent_run)
                return out
            except Exception as e:
                print(e)
                mlclient.set_terminated(parent_run,status='FAILED')
        else:
            return func(self, *args, **kwargs)
            
    return create_mlflow

def mlflow_stage(func):
    
    def create_mlflow_stages(self, *args, **kwargs):
        
        if self.mlflow_flag:
            
            try:
                
                mlclient = MlflowClient()
                parent_run_id = self.parent_run
                exp_id = self.exp_id
                stage_name = kwargs.get('stage_name')
                file_name = kwargs.get('file_name')
                run_tag = DefaultRunContext().tags()  
                run_tag.update({MLFLOW_RUN_NAME:stage_name})
                
                run_tag.update({MLFLOW_PARENT_RUN_ID:parent_run_id})
                stage_run = mlclient.create_run(experiment_id=exp_id,tags=run_tag)
                stage_run_id = stage_run.info.run_id
                
                out = func(self, *args, **kwargs)
                save_artifacts(out, mlclient, stage_run_id, stage_name, file_name)
                
                class_instance = self.operator
                save_params(mlclient, class_instance, stage_run_id)
                
                if self.operator == self.classifier and "train" in stage_name:
                    save_model(mlclient, self.model, stage_run_id, stage_name)
                elif self.operator == self.classifier and 'pred' in stage_name:
                    save_metrics(mlclient, self, stage_run_id)
                    
                mlclient.set_terminated(stage_run_id)
                return out
            
            except Exception as e:
                print(e)
                mlclient.set_terminated(stage_run_id,status='FAILED')
        
        else:
            
            return func(self, *args, **kwargs)
            
    return create_mlflow_stages

def save_artifacts(out, mlclient, stage_run_id, stage_name, filename):
    
    try:
        if filename is not None:
            temp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_dir, filename)  
            with open(f"{tmp_file_path}","w") as file:
                file.write(str(out))
            mlclient.log_artifacts(stage_run_id, temp_dir, artifact_path = stage_name)
        
    except Exception as e:
        print(e)
        
def save_params(mlclient, class_instance, run_id):
    
    try:
        
        init_sign = inspect.signature(class_instance.__init__)
        class_params = {name: getattr(class_instance, name) for name, _ in init_sign.parameters.items() if name != 'self'}  
        
        params_list = [Param(k,str(v)) for k,v in class_params.items()]
        mlclient.log_batch(run_id,params=params_list)
        
    except Exception as e:
        print(e)
    
def save_model(mlclient, model_instance, stage_run_id, stage_name):
    
    try:
        
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, "model.pkl")  
        with open(f"{tmp_file_path}","wb") as pkl:
            pickle.dump(model_instance,pkl)
        mlclient.log_artifacts(stage_run_id, temp_dir, artifact_path = stage_name)
        
    except Exception as e:
        print(e)
        
def save_metrics(mlclient, obj_instance, run_id):
    time_stamp = int(time.time() * 1000)
    metrics_list = [Metric("accuracy_score",obj_instance.accuracy_score,time_stamp,1)]
    mlclient.log_batch(run_id,metrics=metrics_list)
    