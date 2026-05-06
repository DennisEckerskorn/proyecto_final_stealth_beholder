import os

import onnx
import tensorflow as tf
import tf2onnx


def ejecutar_exportacion(
    model_path="models/optimized_model.h5",
    output_path="output/models/modelo_final.onnx",
):
    """
    Convierte el modelo Keras (.h5) a formato ONNX (.onnx).

    Devuelve la ruta del ONNX si la conversión y validación terminan bien; si no,
    devuelve None para que el flujo principal no anuncie una exportación inexistente.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"--- 🧠 Cargando modelo Keras: {model_path} ---")

    if not os.path.exists(model_path):
        print(f"❌ Error crítico: No se encuentra el archivo {model_path}")
        print(f"Directorio actual: {os.getcwd()}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Error al cargar el modelo H5: {e}")
        return None

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

    print("--- 🔄 Convirtiendo a ONNX (Modo Inferencia: training=False) ---")

    @tf.function(input_signature=spec)
    def model_fn(x):
        return model(x, training=False)

    try:
        tf2onnx.convert.from_function(
            model_fn,
            input_signature=spec,
            opset=13,
            output_path=output_path,
        )
    except Exception as e:
        print(f"❌ Error durante la conversión de tf2onnx: {e}")
        return None

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"❌ Error: tf2onnx terminó pero no generó un archivo válido en {output_path}")
        return None

    try:
        check_model = onnx.load(output_path)
        onnx.checker.check_model(check_model)
    except Exception as e:
        print(f"❌ Error de validación: El modelo exportado tiene problemas. Detalle: {e}")
        return None

    print(f"✅ ¡Conversión finalizada y guardada en {output_path}!")
    print(f"🛡️ Verificación exitosa: El archivo ONNX en {output_path} es válido.")
    print("--- Proceso Stealth Beholder (Export) Completado ---")
    return output_path


if __name__ == "__main__":
    ejecutar_exportacion()
