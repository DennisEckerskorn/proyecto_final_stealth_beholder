import tensorflow as tf
import os
from src.datapipeline import load_datasets
from src.model import create_model, compile_model

def train_and_evaluate(data_path="data/dataset", output_path="models/optimized_model.h5"):
    BATCH_SIZE = 16
    EPOCHS = 40

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No existe la ruta del dataset {data_path}."
            "Debe contener las carpetas 'capturas_juego' y 'capturas_real'."
        )

    print("----- ENTRENAMIENTO DEL MODELO -----")

    train_ds, val_ds, class_names = load_datasets(
        data_dir=data_path,
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        validation_split=0.2
    )

    model = create_model()
    model = compile_model(model, learning_rate=0.0001)

    # Definición de Callbacks profesionales
    callbacks = [
        # 1. Parada temprana: Si el error en validación no baja en 5 épocas, se detiene.
        # restore_best_weights asegura que nos quedamos con el mejor modelo, no con el último.
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 2. Reducción de tasa de aprendizaje: Si se estanca, bajamos la velocidad para afinar.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    print("\n" + "="*30)
    print("   RESULTADOS FINALES")
    print("="*30)

    # Evaluación con el mejor estado del modelo recuperado por EarlyStopping
    metrics = model.evaluate(val_ds, verbose=0)
    acc = metrics[1]

    print(f"\n✅ Entrenamiento finalizado.")
    print(f"✅ Precisión en Validación (Mejor época): {acc * 100:.2f}%\n")

    # Guardamos el modelo optimizado en la ruta configurada.
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_path)
    print(f"💾 Modelo guardado en: {output_path}")

if __name__ == "__main__":
    train_and_evaluate()