from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os,sys 
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


class ModelTrainer:


    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self, x, y):
        try:
            param_grid = {
                'max_depth': [2, 3, 5],
                'learning_rate': [0.01, 0.001],
                'n_estimators': [50, 100],
            }
            xgb_model = XGBClassifier()
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=5,
                n_jobs=1,  # decrease number of parallel jobs
                verbose=2,
            )
            n_samples = len(x)
            if n_samples > 10000:
                x_subset = x[:10000]  # use a subset of data if more than 10,000 samples
                y_subset = y[:10000]
            else:
                x_subset = x
                y_subset = y
            grid_search.fit(x_subset, y_subset)
            best_params = grid_search.best_params_
            logging.info(f"Best Tuned Parameters {best_params}")
            return best_params
        except Exception as e:
            raise SensorException(e, sys)
    
    def train_model(self,x,y, best_params):
        try:
            #best_params = self.fine_tune(x, y)
            xgb_clf =  XGBClassifier(**best_params)
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Fine tuning the model")
            best_params = self.fine_tune(x_train, y_train)

            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train,best_params=best_params)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score  =f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score  =f1_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")
            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)