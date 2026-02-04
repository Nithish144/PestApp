# Android Pest Detection APK (Offline, CameraX + TFLite)

This guide focuses **only on inference + mobile APK logic** for an Android app that performs **single-pest classification** and **multi-pest detection** fully offline.

## 1) Overall APK Architecture (Activities, Flow)

### High-level screens
1. **HomeActivity** (or MainActivity)
   - Two cards/tabs/buttons:
     - **Single Pest Classification**
     - **Multiple Pest Detection**
   - Entry to camera/gallery pick.

2. **CaptureActivity** (shared for both modes)
   - CameraX Preview + capture button.
   - “Gallery” button for image picker.
   - When image is selected/captured, navigate to **ResultActivity** with mode + URI/Bitmap.

3. **ResultActivity**
   - If **Classification mode**:
     - Display top-1 class name (bee/caterpillar/grasshopper) + confidence %.
   - If **Detection mode**:
     - Draw bounding boxes and labels over image.
     - Show count per class and total count.

### Suggested module/package structure
```
app/
  src/main/java/com/yourapp/
    ui/
      HomeActivity.kt
      CaptureActivity.kt
      ResultActivity.kt
      overlay/
        DetectionOverlayView.kt
    data/
      ModelRepository.kt
      PestLabels.kt
    ml/
      ClassificationTFLite.kt
      DetectionTFLite.kt
      ImagePreprocessor.kt
      YoloPostProcessor.kt
```

### Flow
1. **HomeActivity** → user chooses mode.
2. **CaptureActivity** uses CameraX or gallery picker.
3. **ResultActivity** loads appropriate model and runs inference:
   - **Classification** → softmax → label + confidence.
   - **Detection** → YOLO post-processing → NMS → boxes/labels.

---

## 2) Model Conversion Steps

### 2.1 Convert classification model: `.h5 → .tflite`
**Goal:** TFLite model for on-device inference.

**Python conversion (TensorFlow 2.x):**
```python
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("best_classification_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional

# (Optional) for full-integer quantization with representative dataset:
# converter.representative_dataset = rep_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open("pest_classification.tflite", "wb") as f:
    f.write(tflite_model)
```

### 2.2 Convert YOLOv8: `.pt → .tflite` (preferred) or ONNX
**Goal:** Android-compatible model (TFLite or ONNX) with post-processing.

#### Option A: YOLOv8 → TFLite (recommended)
Using Ultralytics (exporting to TFLite):
```bash
yolo export model=yolov8_best.pt format=tflite imgsz=640
```
Output: `yolov8_best.tflite` (may include metadata and fused ops).

For quantized TFLite:
```bash
yolo export model=yolov8_best.pt format=tflite imgsz=640 int8
```
> You may need a representative dataset for better INT8 accuracy.

#### Option B: YOLOv8 → ONNX (fallback)
```bash
yolo export model=yolov8_best.pt format=onnx imgsz=640
```
Then use ONNX Runtime Mobile on Android.

**Recommendation:** Use TFLite unless you need advanced ops not supported.

---

## 3) Android-side Inference Logic

### 3.1 Common image preprocessing
- Convert camera image to `Bitmap`.
- Resize to model input size (e.g., 224x224 for classification, 640x640 for YOLO).
- Normalize pixel values:
  - For float models: `input = (pixel / 255.0)`
  - For INT8/UINT8: follow quantization scale/zero-point.

### 3.2 Classification (Single pest)
- Input: `1 x H x W x 3` float tensor.
- Output: vector of class scores → softmax → highest probability.
- Labels: `{bee, caterpillar, grasshopper}`.

### 3.3 Detection (YOLOv8)
- Input: `1 x 640 x 640 x 3` float tensor.
- Output: YOLO head tensor (e.g., `[1, N, 6]` or `[1, 84, 8400]` depending on export).
- Post-processing:
  1. Parse boxes + objectness + class scores.
  2. Apply confidence threshold.
  3. Apply **NMS (Non-Max Suppression)** to filter overlaps.
  4. Map boxes back to original image size.

---

## 4) Pseudo-code + Sample Kotlin Code

### 4.1 Pseudo-code (shared)
```
if mode == CLASSIFICATION:
    bitmap = loadImage()
    input = preprocess(bitmap, size=224)
    scores = classifier.run(input)
    label, conf = argmax(scores)
    show(label, conf)

if mode == DETECTION:
    bitmap = loadImage()
    input = preprocess(bitmap, size=640)
    raw = yolo.run(input)
    detections = postprocess_yolo(raw, conf=0.25, nms=0.45)
    show_boxes(bitmap, detections)
```

### 4.2 Kotlin: Model loading (TFLite)
```kotlin
class TFLiteLoader(private val context: Context) {
    fun loadModel(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
```

### 4.3 Kotlin: Classification inference
```kotlin
class ClassificationTFLite(
    context: Context,
    modelName: String,
    private val labels: List<String>
) {
    private val interpreter: Interpreter

    init {
        val modelBuffer = TFLiteLoader(context).loadModel(modelName)
        interpreter = Interpreter(modelBuffer)
    }

    fun predict(bitmap: Bitmap): Pair<String, Float> {
        val inputSize = 224
        val input = preprocess(bitmap, inputSize, inputSize)
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)

        val probs = softmax(output[0])
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        return labels[maxIdx] to probs[maxIdx]
    }
}
```

### 4.4 Kotlin: YOLOv8 detection inference (TFLite)
```kotlin
data class Detection(
    val label: String,
    val confidence: Float,
    val rect: RectF
)

class YoloV8TFLite(
    context: Context,
    modelName: String,
    private val labels: List<String>
) {
    private val interpreter: Interpreter

    init {
        val modelBuffer = TFLiteLoader(context).loadModel(modelName)
        interpreter = Interpreter(modelBuffer)
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val inputSize = 640
        val input = preprocess(bitmap, inputSize, inputSize)

        // Example output shape: [1, 84, 8400] for YOLOv8
        val output = Array(1) { Array(84) { FloatArray(8400) } }

        interpreter.run(input, output)

        val detections = parseYoloOutput(output, labels)
        return nms(detections, iouThreshold = 0.45f, confThreshold = 0.25f)
    }
}
```

### 4.5 Kotlin: Preprocessing + utilities
```kotlin
fun preprocess(bitmap: Bitmap, width: Int, height: Int): Array<Array<Array<FloatArray>>> {
    val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
    val input = Array(1) { Array(height) { Array(width) { FloatArray(3) } } }

    for (y in 0 until height) {
        for (x in 0 until width) {
            val pixel = resized.getPixel(x, y)
            input[0][y][x][0] = ((pixel shr 16) and 0xFF) / 255f
            input[0][y][x][1] = ((pixel shr 8) and 0xFF) / 255f
            input[0][y][x][2] = (pixel and 0xFF) / 255f
        }
    }
    return input
}

fun softmax(logits: FloatArray): FloatArray {
    val max = logits.maxOrNull() ?: 0f
    val exp = logits.map { kotlin.math.exp(it - max) }
    val sum = exp.sum()
    return exp.map { (it / sum).toFloat() }.toFloatArray()
}
```

---

## 5) Final Year Project Explanation (Clear & Concise)

- The APK runs fully **offline** using **TensorFlow Lite** models.
- There are two modes:
  - **Single Pest Classification:** predicts the pest class for one insect in an image.
  - **Multiple Pest Detection:** locates many pests and draws bounding boxes with labels.
- The app uses **CameraX** for capturing images and also supports gallery images.
- The inference pipeline includes:
  1. Image preprocessing (resize + normalization).
  2. Running the model locally using TFLite interpreter.
  3. Post-processing (softmax for classification, NMS for detection).
  4. Displaying the results on screen.

This design ensures **low-latency**, **privacy-preserving** pest detection without network connectivity.
