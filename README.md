# ContentRecomenderApp

Sistema híbrido de recomendación de películas basado en **Apache Spark**, combinando **Filtrado Colaborativo (ALS)** y **Filtrado Basado en Contenido**, utilizando el dataset MovieLens 100K.

## Tecnologías utilizadas

* Python
* Apache Spark (PySpark)
* MovieLens 100K dataset

---

## Objetivo del proyecto

El objetivo de este proyecto es construir un sistema de recomendación capaz de predecir películas relevantes para cada usuario combinando dos enfoques:

* **Filtrado colaborativo (ALS)**: aprende patrones de comportamiento entre usuarios y películas.
* **Filtrado basado en contenido**: utiliza los géneros de las películas para capturar similitud semántica.

La combinación de ambos modelos permite mejorar la calidad de las recomendaciones y mitigar limitaciones como el *cold start*.

---

## Enfoque del sistema

El sistema se compone de tres etapas principales:

### 1. Preparación de datos

* Carga del dataset MovieLens 100K
* Transformación de géneros en variables explotables
* Construcción de perfiles de usuario basados en sus valoraciones

### 2. Modelos de recomendación

* **ALS (Alternating Least Squares)** entrenado con PySpark MLlib
* **Modelo basado en contenido** usando similitud entre preferencias de usuario y géneros de películas

### 3. Modelo híbrido

Se combinan ambos enfoques mediante una ponderación:

* 70% ALS
* 30% contenido

Esto permite equilibrar patrones globales de comportamiento con preferencias individuales.

---

## Evaluación

El modelo se evalúa utilizando **Precision@K**, midiendo la proporción de elementos relevantes dentro de las K recomendaciones principales por usuario.

---

## Decisiones de diseño

* Se utiliza MovieLens 100K por su equilibrio entre simplicidad y estructura realista.
* Se descartan modelos de Deep Learning como Autoencoders o Neural Collaborative Filtering debido a:

  * Tamaño reducido del dataset
  * Mayor coste computacional
  * Complejidad innecesaria para el objetivo del proyecto
* Se prioriza una solución escalable y explicable basada en Spark.

---

## Limitaciones

* Problema de *cold start* en usuarios con pocas valoraciones.
* Sensibilidad a la distribución del dataset.
* Dependencia de géneros como única fuente de contenido (modelo contenido simplificado).

---

## Conclusión

Este proyecto implementa un sistema híbrido funcional y escalable de recomendación, combinando técnicas clásicas de filtrado colaborativo y basado en contenido sobre Apache Spark.
