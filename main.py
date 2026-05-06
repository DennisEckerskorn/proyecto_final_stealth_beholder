import os
from pathlib import Path

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.config.optimizer.set_jit(False)

from src.datapipeline import CLASS_NAMES
from src.export_onnx import ejecutar_exportacion
from src.predict import ejecutar_prediccion
from src.train import train_and_evaluate


DATA_DIR = Path(os.getenv("DATA_DIR", "data/dataset"))
PATH_H5 = Path(os.getenv("PATH_H5", "models/optimized_model.h5"))
PATH_ONNX = Path(os.getenv("PATH_ONNX", "output/models/modelo_final.onnx"))
TEST_IMAGE = os.getenv("TEST_IMAGE")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_dataset_images(data_dir: Path):
    """Devuelve todas las imágenes del dataset completo, ordenadas por clase y nombre."""
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield image_path, class_name


def _run_single_prediction(image_path: Path, model_path: Path) -> None:
    print(f"\n[TEST FINAL] Analizando: {image_path}")
    resultado, score = ejecutar_prediccion(str(image_path), str(model_path))
    if score is None:
        print(f"Resultado: {resultado}")
    else:
        print(f"Resultado: {resultado} {score:.2f}%")


def _run_dataset_predictions(data_dir: Path, model_path: Path) -> None:
    images = list(_iter_dataset_images(data_dir))
    if not images:
        print(f"❌ No se encontraron imágenes en el dataset {data_dir}")
        return

    print(f"\n[TEST DATASET] Analizando dataset completo: {data_dir}")
    print(f"Imágenes encontradas: {len(images)}")

    correct = 0
    failed = 0

    for image_path, expected_class in images:
        resultado, score = ejecutar_prediccion(str(image_path), str(model_path))
        if score is None:
            failed += 1
            print(f"❌ {image_path} -> {resultado}")
            continue

        is_correct = resultado == expected_class
        correct += int(is_correct)
        marker = "✅" if is_correct else "❌"
        print(
            f"{marker} {image_path} | esperado={expected_class} "
            f"predicho={resultado} confianza={score:.2f}%"
        )

    evaluated = len(images) - failed
    print("\n" + "=" * 40)
    print("   RESUMEN DE PREDICCIÓN DEL DATASET")
    print("=" * 40)
    print(f"Imágenes totales: {len(images)}")
    print(f"Evaluadas correctamente por ONNX: {evaluated}")
    print(f"Errores de inferencia: {failed}")
    if evaluated:
        print(f"Aciertos: {correct}/{evaluated} ({(correct / evaluated) * 100:.2f}%)")


def main():
    print("--- 🛡️ SISTEMA STEALTH BEHOLDER ---")

    # FASE 1: Entrenamiento
    if not PATH_H5.exists():
        train_and_evaluate(data_path=str(DATA_DIR), output_path=str(PATH_H5))
    else:
        print(f"✅ Modelo Keras existente encontrado en: {PATH_H5}")

    # FASE 2: Exportación
    exported_path = ejecutar_exportacion(model_path=str(PATH_H5), output_path=str(PATH_ONNX))
    if exported_path is None:
        print("❌ No se puede continuar porque la exportación a ONNX falló.")
        return

    onnx_path = Path(exported_path)

    # FASE 3: Predicción
    if TEST_IMAGE:
        test_image_path = Path(TEST_IMAGE)
        if test_image_path.exists():
            _run_single_prediction(test_image_path, onnx_path)
        else:
            print(f"❌ No se encontró la imagen de prueba en {test_image_path}")
    else:
        _run_dataset_predictions(DATA_DIR, onnx_path)


if __name__ == "__main__":
    main()
