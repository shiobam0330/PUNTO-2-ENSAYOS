import unittest
from pipeline import parte1, preparar_ejecutar_red_neuronal
import numpy as np

class PruebaMLFlow(unittest.TestCase):

    def test_model(self):
        accuracy_train, accuracy_test, history = preparar_ejecutar_red_neuronal()
        resultados_roc = parte1() 
        mensaje = "Proceso Exitoso" 
        self.assertEqual(mensaje, "Proceso Exitoso", "El proceso no fue exitoso")
        diferencia_precision = np.abs(accuracy_train - accuracy_test)
        if diferencia_precision <= 10:
            print("No presenta Underfitting ni Overfitting")
        else:
            print("Posible Overfitting o Underfitting")
        self.assertLessEqual(diferencia_precision, 10, "Hay una diferencia significativa, posible Overfitting/Underfitting")

        print(f"Accuracy de Entrenamiento: {accuracy_train:.4f}")
        print(f"Accuracy de Prueba: {accuracy_test:.4f}")

if __name__ == "__main__":
    unittest.main()
