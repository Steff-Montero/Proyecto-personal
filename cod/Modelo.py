import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

class EstimacionBootstrapping:
    def __init__(self, model = LogisticRegression(random_state = 0, max_iter = 200), data = 'titanic', caract_entrenamiento = None,
                 caract_prueba = None, result_entrenamiento = None, result_prueba = None):
        self.model = model
        self.data = sns.load_dataset(data)
        self.__caract_entrenamiento = caract_entrenamiento
        self.__caract_prueba = caract_prueba
        self.__result_entrenamiento = result_entrenamiento
        self.__result_prueba = result_prueba
        self.__precision = []
    
    def get_model(self):
        return self.model
    
    def set_model(self, new_model):
        self.model = new_model
    
    def get_data(self):
        return self.data
    
    def set_data(self, new_data):
        self.data = sns.load_dataset(new_data)
        
    def imputar(self, variables = ['survived', 'age', 'pclass', 'fare']):
        self.data = self.data[variables].dropna()

    def categorica_a_numerica(self, metodo = 'label'):
        if metodo == 'label':
            le = LabelEncoder()
            for col in self.data.select_dtypes(include = ['object', 'category']).columns:
                self.data[col] = le.fit_transform(self.data[col])
        elif metodo == 'onehot':
            self.data = pd.get_dummies(self.data, drop_first = True)
        else:
            raise ValueError("El método debe ser 'label' o 'onehot'.")
        
    def dividir_datos(self, variables_descriptivas = ['age', 'pclass', 'fare'], variable_objetivo = 'survived'):
        caracteristicas = self.data[variables_descriptivas]
        objetivo = self.data[variable_objetivo]
        self.__caract_entrenamiento, self.__caract_prueba, self.__result_entrenamiento, self.__result_prueba = train_test_split(caracteristicas,
                                                                                                                        objetivo,
                                                                                                                        train_size = 0.7,
                                                                                                                        random_state = 42)
    def entrenar_modelo(self):
        self.model.fit(self.__caract_entrenamiento, self.__result_entrenamiento.values.ravel())
        
    def proyeccion(self):
        result_predic_prueba = self.model.predict(self.__caract_prueba)
        precision_prueba = accuracy_score(self.__result_prueba, result_predic_prueba)
        print(f"Precisión en el conjunto de prueba: {precision_prueba: .2f}")
        
        resultado_comparacion = pd.DataFrame({'Real': self.__result_prueba.values.ravel(), 'Predicción': result_predic_prueba})
        print(resultado_comparacion.head(10))
        print(confusion_matrix(self.__result_prueba, result_predic_prueba))
        print(classification_report(self.__result_prueba, result_predic_prueba))
        
    def evaluacion_bootstrap(self, iteraciones = 1000):
        for i in range(iteraciones):
            caracteristicas_bs, resultados_bs = resample(self.__caract_entrenamiento, self.__result_entrenamiento.values.ravel(), replace = True)
            result_pred = self.model.predict(caracteristicas_bs)
            puntaje = accuracy_score(resultados_bs, result_pred)
            self.__precision.append(puntaje)
            
        sns.kdeplot(self.__precision)
        plt.title('Precisión en 1000 muestras bootstrap del conjunto de prueba separado')
        plt.xlabel('Precisión')
        plt.show()
        
        mediana = np.percentile(self.__precision, 50)
        alpha = 5
        interval_confi_inf = np.percentile(self.__precision, alpha / 2)
        interval_confi_sup = np.percentile(self.__precision, 100 - alpha / 2)
        
        print(f"La precisión del modelo se informa en el conjunto de prueba. Se utilizaron"
              f"1000 muestras de bootstrap para calcular los intervalos de confianza al 95%. \n"
              f"La precisión mediana es {mediana: .2f} con un intervalo de confianza al 95% de "
              f"[{interval_confi_inf: .2f}, {interval_confi_sup: .2f}]")
        
        sns.kdeplot(self.__precision)
        plt.xlabel('Precisión')
        plt.axvline(mediana, 0, 14, linestyle = '--', color = 'red')
        plt.axvline(interval_confi_inf, 0, 14, linestyle = '--', color = 'red')
        plt.axvline(interval_confi_sup, 0, 14, linestyle = '--', color = 'red')
        plt.show()