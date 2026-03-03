import joblib

import Definitions

import os.path as osp

from src.TextProcessing import processing_text

class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")

        self.model_path = osp.join(self.model_path, "best_svc.joblib")

        self.model = joblib.load(self.model_path)
        # Inicializar variables
        self.input_df = ""
        # Clase de preprocesamiento de la información


    def load_input_data(self, input_data):
        print("ModelController.load_input_data ->")
        try:
            input_text = input_data.strip()
            if not input_text:
                return None, False


            return input_text, True

        except:
            raise("Ocurrió un error al leer la información de entrada")

    def predict(self, text: str):
        print("ModelController.predict ->")

        try:
            # Convert text into vector form
            X = processing_text(text)

            y_pred = self.model.predict([X])

            return text, y_pred[0]

        except Exception as e:
            raise Exception(f"Error during prediction: {e}")

