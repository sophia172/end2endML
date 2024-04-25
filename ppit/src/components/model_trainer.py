import sys
from ppit.src.exception import CustomException
from ppit.src.logger import logging
from ppit.src.utils import writer, evaluate_models
import os
from sklearn.ensemble import (
    RandomForestRegressor,
                                )
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score




class ModelTrainer:
    def __init__(self):
        pass

    def __call__(self, X_train, X_test, y_train, y_test):
        try:
            logging.info(f"Import data")
            # TODO
        except CustomException as e:
            logging.error(e)
            raise CustomException(e, sys)


class BaselineSearch:
    def __init__(self):
        self.save_path = os.path.join("model", "baseline")
        os.makedirs(self.save_path, exist_ok=True)

    def __call__(self, X_train, X_test,  y_train, y_test):
        try:
            logging.info(f"Import data")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                # "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBClassifier": XGBRegressor(),
                # "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                # "AdaBoost Classifier": AdaBoostRegressor(),
            }


            model_report, model_obj = evaluate_models(X_train, X_test, y_train, y_test, models=models)
            logging.info(f"Finished training models")
            ## Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## Get best model name
            baseline_name: str = [k for k, v in model_report.items() if v == best_model_score][0]
            baseline_model: object = model_obj[baseline_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")#
            writer(baseline_model, os.path.join(self.save_path, baseline_name + '.p'))

            r2_square = r2_score(y_test, baseline_model.predict(X_test))
            logging.info(f"Baseline model found as {baseline_name} with r2_score of {r2_square}")

            return r2_square

        except CustomException as e:
            logging.error(e)
            raise CustomException(e, sys)


if __name__=="__main__":
    from ppit.src.components.data_ingestion import DataLoader
    from sklearn.model_selection import train_test_split
    from ppit.src.components.models import CNN
    data_loader = DataLoader("../../data/vertexAI_PPIT_data.csv")

    data = data_loader()
    X_train, X_test, y_train, y_test = train_test_split(*data)
    model = CNN("../../config/model_params_example.yml")
    model.build()

    model.debug_compile_fit(X_train, X_test, y_train, y_test)
    # model.compile()
    # model.fit(X_train, y_train, X_test, y_test)
    # model.save()