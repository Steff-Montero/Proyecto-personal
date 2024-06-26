
# Proyecto Personal

## Estimación de Error con Bootstrapping

Un script de Python que utiliza bootstrapping para estimar el error de un modelo de regresión dado, utilizando el dataset elegido por el usuario de la librería Seaborn.

## Descripción

`EstimacionBootstrapping` es una clase que facilita el proceso de limpiar datos, transformar variables categóricas en numéricas, dividir los datos en conjuntos de entrenamiento y prueba, entrenar un modelo de regresión logística, y estimar el error del modelo utilizando bootstrapping.

## Instalación

### Requisitos previos

- Python 3.6 o superior
- Las siguientes librerías de Python:
  - matplotlib
  - pandas
  - numpy
  - scikit-learn
  - seaborn

### Instrucciones

1. Clona el repositorio:
    ```bash
    git clone https://github.com/Steff-Montero/Proyecto-personal.git

    ```
2. Navega al directorio del proyecto:
    ```bash
    cd Proyecto-personal
    ```
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

A continuación se muestra un ejemplo de cómo usar la clase `EstimacionBootstrapping`:

```python
from Modelo import EstimacionBootstrapping

# Inicializa el objeto con el dataset de Titanic
estimador = EstimacionBootstrapping()

# Limpia los datos
estimador.limpieza(variables=['survived', 'age', 'pclass', 'fare'])

# Convierte variables categóricas a numéricas
estimador.categorica_a_numerica(metodo='label')

# Divide los datos en entrenamiento y prueba
estimador.dividir_datos(variables_descriptivas=['age', 'pclass', 'fare'], variable_objetivo='survived')

# Entrena el modelo
estimador.entrenar_modelo()

# Realiza predicciones y muestra los resultados
estimador.proyeccion()

# Evalúa el modelo usando bootstrapping
estimador.evaluacion_bootstrap(iteraciones=1000)
```

## Características
- Elimina filas con valores nulos
- Convierte variables categóricas a numéricas usando label encoding o one-hot encoding
- Divide los datos en conjuntos de entrenamiento y prueba
- Entrena un modelo de regresión logística
- Realiza predicciones y muestra resultados de precisión, matriz de confusión e informe de clasificación
- Estima el error del modelo usando bootstrapping

## Contribución
Para contribuir, por favor sigue las siguientes instrucciones:

- Haz un fork del repositorio
- Crea una nueva rama (git checkout -b feature/nueva-caracteristica)
- Realiza tus cambios y haz commit (git commit -am 'Añadir nueva característica')
- Haz push a la rama (git push origin feature/nueva-caracteristica)
- Abre un Pull Request

### Tecnologías
![Python](https://img.shields.io/badge/-Python-0066CC)
![JupyterNotebook](https://img.shields.io/badge/-JupyterNotebook-FF8000)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - mira el archivo [LICENSE](LICENSE) para más detalles.

## Autor

- [@Steff-Montero](https://github.com/Steff-Montero)

## Referencias
- [Documentación de pandas](https://pandas.pydata.org/docs/)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Documentación Matplotlip.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
- [Documentación de Seaborn](https://seaborn.pydata.org/)
- [Bootstrapping - Machine Learning](https://carpentries-incubator.github.io/machine-learning-novice-python/07-bootstrapping/index.html)
- [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

Este README proporciona una guía completa y clara sobre cómo usar la clase `EstimacionBootstrapping`, asegurando que cualquier usuario pueda entender rápidamente cómo funciona el script y cómo aplicarlo a sus propios datos.
