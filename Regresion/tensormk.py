# ==============================
# IMPORTAR LIBRERÍAS
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# GENERAR DATOS SINTÉTICOS PARA REGRESIÓN
# ==============================
# Crear un dataset donde las imágenes son patrones 2D
# y el target es un valor CONTINUO (intensidad de píxeles combinada)

np.random.seed(42)

# Generar 5000 imágenes 2D aleatorias (28x28)
n_samples = 5000
X_full = np.random.rand(n_samples, 28, 28).astype("float32")

# El target será un valor continuo basado en característica de las imágenes
# Regla: suma de región central + ruido (valor continuo realista)
y_full = np.array([
    X_full[i, 10:20, 10:20].sum() + np.random.normal(0, 0.5)
    for i in range(n_samples)
]).astype("float32")

# Normalizar target a rango [0, 1]
y_full = (y_full - y_full.min()) / (y_full.max() - y_full.min())

# Dividir en train (60%), validation (20%), test (20%)
split_train = int(0.6 * len(X_full))
split_val = int(0.8 * len(X_full))

X_train = X_full[:split_train]
y_train = y_full[:split_train]

X_val = X_full[split_train:split_val]
y_val = y_full[split_train:split_val]

X_test = X_full[split_val:]
y_test = y_full[split_val:]

# ==============================
# NORMALIZAR DATOS
# ==============================

# Imágenes ya están en [0, 1], solo agregar dimensión canal
X_train = X_train.reshape(-1, 28, 28, 1)
X_val   = X_val.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# ==============================
# DEFINIR ARQUITECTURA CNN PARA REGRESIÓN
# ==============================
# JUSTIFICACIÓN DE LA ARQUITECTURA:
# - Conv2D: Extrae características locales (bordes, texturas)
# - MaxPooling2D: Reduce dimensionalidad, mantiene características importantes
# - BatchNormalization: Estabiliza gradientes, acelera convergencia
# - Dropout: Regularización, previene overfitting
# - Flatten + Dense: Convierte características en predicción continua
# - Activación final LINEAL: Necesaria para valores continuos en regresión

input_shape = X_train.shape[1:]

model = models.Sequential(name="CNN_Regresion")

# ---- Input Layer ----
model.add(layers.Input(shape=input_shape))

# ---- Bloque Convolucional 1 ----
# 32 filtros extraen características simples (bordes, esquinas)
model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_1'))
model.add(layers.BatchNormalization(name='batch_norm_1'))  # Regularización
model.add(layers.MaxPooling2D((2,2), name='maxpool_1'))    # Reduce tamaño (28→14)

# ---- Bloque Convolucional 2 ----
# 64 filtros capturan características más complejas
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_2'))
model.add(layers.BatchNormalization(name='batch_norm_2'))
model.add(layers.MaxPooling2D((2,2), name='maxpool_2'))    # Reduce (14→7)

# ---- Bloque Convolucional 3 ----
# 128 filtros detectan patrones de alto nivel
model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_3'))
model.add(layers.BatchNormalization(name='batch_norm_3'))
model.add(layers.MaxPooling2D((2,2), name='maxpool_3'))    # Reduce (7→3)

# ---- Aplanar (Flatten) ----
# Convierte matriz 3D en vector 1D para capas densas
model.add(layers.Flatten(name='flatten'))

# ---- Capas Densas con Dropout ----
# 128 neuronas con ReLU para aprender relaciones complejas
model.add(layers.Dense(128, activation='relu', name='dense_1'))
model.add(layers.Dropout(0.5, name='dropout_1'))  # 50% dropout: regularización fuerte

# 64 neuronas, dropout más suave
model.add(layers.Dense(64, activation='relu', name='dense_2'))
model.add(layers.Dropout(0.3, name='dropout_2'))  # 30% dropout

# ---- Capa de Salida para REGRESIÓN ----
# 1 neurona con activación LINEAL
# (Crucialmente diferente de softmax o sigmoid para regresión)
model.add(layers.Dense(1, activation='linear', name='output'))

print("\n" + "="*60)
print("ARQUITECTURA DE LA RED NEURONAL CNN PARA REGRESIÓN")
print("="*60)

# ==============================
# COMPILACIÓN DEL MODELO
# ==============================
# Función de Pérdida: MSE (Mean Squared Error)
#   → Adecuada para REGRESIÓN
#   → Penaliza errores grandes más que pequeños
#   → Derivada suave para backpropagation
#
# Optimizador: Adam
#   → Combina momentum y RMSprop
#   → Buena convergencia en la mayoría de problemas
#   → Adapta tasa de aprendizaje por parámetro
#
# Métrica: MAE (Mean Absolute Error)
#   → Interpretable: error promedio en mismas unidades
#   → Robusta a outliers

model.compile(
    optimizer='adam',
    loss='mse',      # MSE para regresión
    metrics=['mae']  # MAE como métrica de evaluación
)

model.summary()

print("="*60)
print("Optimizador: Adam (adaptive learning rate)")
print("Función de Pérdida: MSE (Mean Squared Error)")
print("Métrica de Evaluación: MAE (Mean Absolute Error)")
print("="*60 + "\n")

# ==============================
# ENTRENAMIENTO
# ==============================
# Parámetros de entrenamiento:
# - epochs=30: Recorre el dataset 30 veces
# - batch_size=32: Procesa 32 muestras antes de actualizar pesos
# - validation_data: Monitorea desempeño en datos no vistos (detección de overfitting)

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60)
print(f"Datos Training: {X_train.shape}")
print(f"Datos Validation: {X_val.shape}")
print(f"Datos Test: {X_test.shape}")
print(f"Épocas: 30 | Batch Size: 32")
print("="*60 + "\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

# ==============================
# EVALUACIÓN
# ==============================

def evaluar(y_real, y_pred, nombre):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)

    print(f"\n{'='*50}")
    print(f"RESULTADOS EN {nombre}")
    print(f"{'='*50}")
    print(f"MAE  (Mean Absolute Error)        : {mae:.6f}")
    print(f"RMSE (Root Mean Squared Error)    : {rmse:.6f}")
    print(f"R²   (Coeficiente de Determinación): {r2:.6f}")
    
    return mae, rmse, r2

# Predicciones
print("\n" + "="*50)
print("PREDICCIONES DEL MODELO")
print("="*50)
y_train_pred = model.predict(X_train, verbose=0)
y_val_pred   = model.predict(X_val, verbose=0)
y_test_pred  = model.predict(X_test, verbose=0)

# Aplanar predicciones (remove extra dimension)
y_train_pred = y_train_pred.flatten()
y_val_pred = y_val_pred.flatten()
y_test_pred = y_test_pred.flatten()

mae_train, rmse_train, r2_train = evaluar(y_train, y_train_pred, "TRAINING")
mae_val, rmse_val, r2_val = evaluar(y_val, y_val_pred, "VALIDATION")
mae_test, rmse_test, r2_test = evaluar(y_test, y_test_pred, "TEST")

# ==============================
# ANÁLISIS DE OVERFITTING / UNDERFITTING
# ==============================

print("\n" + "="*50)
print("ANÁLISIS DE OVERFITTING / UNDERFITTING")
print("="*50)

# Obtener pérdida final de train y validation
train_loss_final = history.history['loss'][-1]
val_loss_final = history.history['val_loss'][-1]

print(f"\nPérdida MSE FINAL:")
print(f"  Training:   {train_loss_final:.6f}")
print(f"  Validation: {val_loss_final:.6f}")

# Calcular diferencia relativa
diferencia_rel = (val_loss_final - train_loss_final) / train_loss_final * 100

print(f"\nDiferencia relativa: {diferencia_rel:.2f}%")

# Diagnóstico
print(f"\n{'─'*50}")
if val_loss_final > train_loss_final * 1.3:
    print("⚠️  DIAGNÓSTICO: HAY OVERFITTING")
    print("    → El modelo memoriza datos de entrenamiento")
    print("    → Generaliza mal en datos nuevos")
    print("    → Recomendación: Aumentar Dropout, menos épocas")
elif rmse_train > 0.1 and rmse_val > 0.1:
    print("⚠️  DIAGNÓSTICO: HAY UNDERFITTING")
    print("    → El modelo no aprende patrones suficientes")
    print("    → Recomendación: Aumentar capas, más épocas")
else:
    print("✅ DIAGNÓSTICO: BUEN AJUSTE")
    print("    → El modelo aprende y generaliza correctamente")
    print("    → MSE train y val son similares")
    print("    → R² indica buen desempeño")

print(f"{'─'*50}\n")

# ==============================
# CURVAS DE PÉRDIDA Y ANÁLISIS
# ==============================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica 1: Pérdida (MSE)
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
axes[0].set_title('Evolución de Pérdida (MSE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfica 2: MAE
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2, marker='o', markersize=3)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2, marker='s', markersize=3)
axes[1].set_title('Evolución de MAE', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
print("✅ Gráficas guardadas en 'curvas_entrenamiento.png'")
plt.show()

# ==============================
# ANÁLISIS VISUAL: Predicciones vs Real
# ==============================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Valores Reales')
axes[0].set_ylabel('Predicciones')
axes[0].set_title(f'TRAINING (R²={r2_train:.4f})')
axes[0].grid(True, alpha=0.3)

# Validation
axes[1].scatter(y_val, y_val_pred, alpha=0.5, s=10, color='orange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('Valores Reales')
axes[1].set_ylabel('Predicciones')
axes[1].set_title(f'VALIDATION (R²={r2_val:.4f})')
axes[1].grid(True, alpha=0.3)

# Test
axes[2].scatter(y_test, y_test_pred, alpha=0.5, s=10, color='green')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[2].set_xlabel('Valores Reales')
axes[2].set_ylabel('Predicciones')
axes[2].set_title(f'TEST (R²={r2_test:.4f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predicciones_vs_real.png', dpi=150, bbox_inches='tight')
print("✅ Gráficas guardadas en 'predicciones_vs_real.png'")
plt.show()

# ==============================
# PRUEBA CON MUESTRA ARTIFICIAL
# ==============================
# Requisito:
# - Tomar una imagen de prueba con mismo size/channels del modelo (28,28,1)
# - Ingresarla al modelo CNN entrenado
# - Analizar predicción y cambios visuales (iluminación, escala, orientación)

print("\n" + "="*60)
print("PRUEBA CON MUESTRA ARTIFICIAL")
print("="*60)

# 1) Crear muestra artificial base (28x28) con un patrón claro en la región central
#    Similar a la lógica del target usado en entrenamiento (suma de zona central)
img_base = np.zeros((28, 28), dtype="float32")

# Región central brillante (controla fuertemente la predicción en este dataset)
img_base[10:20, 10:20] = 0.8

# Agregar un poco de textura/ruido suave para mayor realismo
img_base += np.random.normal(loc=0.0, scale=0.03, size=(28, 28)).astype("float32")
img_base = np.clip(img_base, 0.0, 1.0)

# Formato de entrada para CNN: (batch, h, w, c) = (1, 28, 28, 1)
x_base = img_base.reshape(1, 28, 28, 1)

# Predicción base
pred_base = float(model.predict(x_base, verbose=0).flatten()[0])

print(f"\nPredicción base (muestra artificial original): {pred_base:.6f}")
print("Interpretación: valor continuo estimado por el modelo de regresión.")

# 2) Variación A: ILUMINACIÓN (más brillante)
img_bright = np.clip(img_base * 1.25, 0.0, 1.0)
x_bright = img_bright.reshape(1, 28, 28, 1)
pred_bright = float(model.predict(x_bright, verbose=0).flatten()[0])

# 3) Variación B: ESCALA (objeto central más pequeño)
img_small = np.zeros((28, 28), dtype="float32")
img_small[12:18, 12:18] = 0.8  # región central más pequeña
img_small += np.random.normal(loc=0.0, scale=0.03, size=(28, 28)).astype("float32")
img_small = np.clip(img_small, 0.0, 1.0)
x_small = img_small.reshape(1, 28, 28, 1)
pred_small = float(model.predict(x_small, verbose=0).flatten()[0])

# 4) Variación C: ORIENTACIÓN (rotar 90°)
# Nota: en este dataset sintético, la zona central sigue siendo central al rotar, por eso
# el cambio puede ser pequeño.
img_rot = np.rot90(img_base, k=1).copy()
x_rot = img_rot.reshape(1, 28, 28, 1)
pred_rot = float(model.predict(x_rot, verbose=0).flatten()[0])

print("\nResultados de sensibilidad:")
print(f"- Original      : {pred_base:.6f}")
print(f"- Iluminación + : {pred_bright:.6f}  (delta={pred_bright - pred_base:+.6f})")
print(f"- Escala menor  : {pred_small:.6f}  (delta={pred_small - pred_base:+.6f})")
print(f"- Rotación 90°  : {pred_rot:.6f}  (delta={pred_rot - pred_base:+.6f})")

# 5) Análisis automático solicitado
print("\nANÁLISIS DE LA PREDICCIÓN:")
if pred_bright > pred_base:
    print("- Iluminación: Al aumentar brillo, la predicción sube. Tiene sentido para este problema,")
    print("  porque el target depende de intensidad (suma de píxeles en zona central).")
else:
    print("- Iluminación: El cambio no fue el esperado o fue leve; revisar convergencia del modelo.")

if pred_small < pred_base:
    print("- Escala: Al reducir el tamaño del patrón central, la predicción baja. Tiene sentido,")
    print("  porque hay menos intensidad acumulada en la región relevante.")
else:
    print("- Escala: El efecto fue bajo/no esperado; podría requerir más épocas o ajuste de arquitectura.")

print("- Orientación: En este dataset sintético, rotar 90° puede afectar poco si el patrón sigue")
print("  ocupando la zona central. En tareas reales, la orientación suele afectar más si el modelo")
print("  no fue entrenado con data augmentation (rotaciones).")

# 6) Visualización comparativa
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
axs[0].imshow(img_base, cmap='gray', vmin=0, vmax=1)
axs[0].set_title(f"Original\n{pred_base:.4f}")
axs[0].axis("off")

axs[1].imshow(img_bright, cmap='gray', vmin=0, vmax=1)
axs[1].set_title(f"Iluminación+\n{pred_bright:.4f}")
axs[1].axis("off")

axs[2].imshow(img_small, cmap='gray', vmin=0, vmax=1)
axs[2].set_title(f"Escala menor\n{pred_small:.4f}")
axs[2].axis("off")

axs[3].imshow(img_rot, cmap='gray', vmin=0, vmax=1)
axs[3].set_title(f"Rotación 90°\n{pred_rot:.4f}")
axs[3].axis("off")

plt.tight_layout()
plt.savefig("prueba_muestra_artificial.png", dpi=150, bbox_inches='tight')
print("\n✅ Imagen comparativa guardada en 'prueba_muestra_artificial.png'")
plt.show()

# ==============================
# PRUEBA CON IMAGEN PROPIA DEL USUARIO
# ==============================
from PIL import Image

print("\n" + "="*60)
print("PRUEBA CON IMAGEN PROPIA DEL USUARIO")
print("="*60)

ruta_img = input(
    "Ingresa la ruta de tu imagen (ej: C:/Users/USER/Documents/IA/mi_imagen.png) "
    "o presiona ENTER para omitir: "
).strip()

if ruta_img != "":
    try:
        # 1) Cargar imagen y convertir a escala de grises
        img_user = Image.open(ruta_img).convert("L")  # L = grayscale

        # 2) Redimensionar a 28x28 (mismo size del modelo)
        img_user = img_user.resize((28, 28), Image.Resampling.LANCZOS)

        # 3) Convertir a numpy y normalizar [0,1]
        img_user_np = np.array(img_user).astype("float32") / 255.0

        # 4) Formato CNN: (batch, h, w, c) = (1, 28, 28, 1)
        x_user = img_user_np.reshape(1, 28, 28, 1)

        # Predicción base
        pred_user = float(model.predict(x_user, verbose=0).flatten()[0])

        # Variaciones para análisis visual
        # A) Iluminación +
        img_user_bright = np.clip(img_user_np * 1.25, 0.0, 1.0)
        pred_user_bright = float(
            model.predict(img_user_bright.reshape(1, 28, 28, 1), verbose=0).flatten()[0]
        )

        # B) Escala: reducir contenido y centrarlo
        # Tomamos una versión reducida 20x20 y la pegamos al centro en fondo negro 28x28
        img_small_pil = Image.fromarray((img_user_np * 255).astype(np.uint8)).resize((20, 20), Image.Resampling.LANCZOS)
        canvas = np.zeros((28, 28), dtype="float32")
        canvas[4:24, 4:24] = np.array(img_small_pil).astype("float32") / 255.0
        img_user_small = canvas
        pred_user_small = float(
            model.predict(img_user_small.reshape(1, 28, 28, 1), verbose=0).flatten()[0]
        )

        # C) Orientación: rotar 30 grados
        img_rot_pil = Image.fromarray((img_user_np * 255).astype(np.uint8)).rotate(30, resample=Image.Resampling.BICUBIC)
        img_user_rot = np.array(img_rot_pil).astype("float32") / 255.0
        pred_user_rot = float(
            model.predict(img_user_rot.reshape(1, 28, 28, 1), verbose=0).flatten()[0]
        )

        # Mostrar resultados numéricos
        print("\nResultados con imagen del usuario:")
        print(f"- Original      : {pred_user:.6f}")
        print(f"- Iluminación + : {pred_user_bright:.6f}  (delta={pred_user_bright - pred_user:+.6f})")
        print(f"- Escala menor  : {pred_user_small:.6f}  (delta={pred_user_small - pred_user:+.6f})")
        print(f"- Rotación 30°  : {pred_user_rot:.6f}  (delta={pred_user_rot - pred_user:+.6f})")

        # Análisis automático
        print("\nANÁLISIS (imagen propia):")
        if abs(pred_user_bright - pred_user) > 1e-4:
            print("- Iluminación: Sí afecta la predicción. Cambiar brillo altera intensidades de píxeles.")
        else:
            print("- Iluminación: Efecto leve en esta imagen específica.")

        if abs(pred_user_small - pred_user) > 1e-4:
            print("- Escala: Sí afecta la predicción. Cambiar tamaño modifica patrones espaciales.")
        else:
            print("- Escala: Efecto leve en esta imagen específica.")

        if abs(pred_user_rot - pred_user) > 1e-4:
            print("- Orientación: Sí afecta la predicción. Rotar cambia distribución de rasgos.")
        else:
            print("- Orientación: Efecto leve; el modelo parece relativamente robusto a esta rotación.")

        # Visual comparativo
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(img_user_np, cmap='gray', vmin=0, vmax=1)
        axs[0].set_title(f"Original\n{pred_user:.4f}")
        axs[0].axis("off")

        axs[1].imshow(img_user_bright, cmap='gray', vmin=0, vmax=1)
        axs[1].set_title(f"Iluminación+\n{pred_user_bright:.4f}")
        axs[1].axis("off")

        axs[2].imshow(img_user_small, cmap='gray', vmin=0, vmax=1)
        axs[2].set_title(f"Escala menor\n{pred_user_small:.4f}")
        axs[2].axis("off")

        axs[3].imshow(img_user_rot, cmap='gray', vmin=0, vmax=1)
        axs[3].set_title(f"Rotación 30°\n{pred_user_rot:.4f}")
        axs[3].axis("off")

        plt.tight_layout()
        plt.savefig("prueba_imagen_usuario.png", dpi=150, bbox_inches="tight")
        print("\n✅ Imagen comparativa guardada en 'prueba_imagen_usuario.png'")
        plt.show()

    except Exception as e:
        print(f"\n❌ Error al procesar la imagen: {e}")
        print("Verifica la ruta y que el archivo sea una imagen válida (.png, .jpg, etc).")
else:
    print("Se omitió la prueba con imagen propia.")