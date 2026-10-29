package com.dronedet.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Size
import android.view.Gravity
import android.view.ScaleGestureDetector
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Drone Detector: the phone's back camera through a YOLO26 drone model (ONNX, the same export as for the
 * station: input 1x3x640x640 RGB 0..1 letterboxed with grey 114, output 1x300x6 = x1 y1 x2 y2 score class
 * in input pixels, no NMS needed). Pinch to zoom, slider = detection threshold (default 0.40), "Модель…"
 * loads another .onnx from the phone (kept for the next start); long press = back to the built-in ft_v9.
 */
class MainActivity : ComponentActivity() {
    private val size = 640
    private lateinit var preview: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var info: TextView
    private lateinit var thrText: TextView
    private var camera: Camera? = null
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val exec = Executors.newSingleThreadExecutor()
    @Volatile private var session: OrtSession? = null
    @Volatile private var thr = 0.40f
    @Volatile private var modelName = "ft_v9 (вбудована)"
    private val engines = listOf("CPU", "XNNPACK", "NNAPI")
    @Volatile private var engine = "XNNPACK"
    private var modelBytes: ByteArray? = null
    private var fps = 0f
    private var inferMs = 0f
    private var lastT = 0L

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else info.text = "Немає дозволу на камеру"
    }
    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) loadPicked(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()
        val prefs = getPreferences(MODE_PRIVATE)
        thr = prefs.getFloat("thr", 0.40f)
        engine = prefs.getString("engine", "XNNPACK")!!
        val custom = File(filesDir, "custom.onnx")
        if (custom.exists()) loadModel(custom.readBytes(), prefs.getString("model", "своя модель")!!)
        else loadModel(assets.open("ft_v9.onnx").use { it.readBytes() }, "ft_v9 (вбудована)")
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera() else askCamera.launch(Manifest.permission.CAMERA)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun buildUi() {
        val root = FrameLayout(this)
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        overlay = OverlayView(this)
        root.addView(preview, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        root.addView(overlay, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(150, 0, 0, 0))
            setPadding(24, 16, 24, 32)
        }
        info = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 14f
            // tap = next inference engine (CPU / XNNPACK / NNAPI), to pick the fastest on this phone
            setOnClickListener {
                engine = engines[(engines.indexOf(engine) + 1) % engines.size]
                getPreferences(MODE_PRIVATE).edit().putString("engine", engine).apply()
                modelBytes?.let { loadModel(it, modelName) }
            }
        }
        thrText = TextView(this).apply { setTextColor(Color.WHITE); textSize = 16f }
        val bar = SeekBar(this).apply {
            max = 90
            progress = ((thr - 0.05f) * 100).roundToInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    thr = 0.05f + p / 100f
                    thrText.text = "Поріг: %.2f".format(thr)
                    getPreferences(MODE_PRIVATE).edit().putFloat("thr", thr).apply()
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }
        thrText.text = "Поріг: %.2f".format(thr)
        val btn = Button(this).apply {
            text = "Модель…"
            setOnClickListener { pickModel.launch(arrayOf("*/*")) }
            setOnLongClickListener {
                File(filesDir, "custom.onnx").delete()
                loadModel(assets.open("ft_v9.onnx").use { it.readBytes() }, "ft_v9 (вбудована)")
                true
            }
        }
        panel.addView(info)
        panel.addView(thrText)
        panel.addView(bar)
        panel.addView(btn, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        root.addView(panel, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
        setContentView(root)

        val pinch = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                val cam = camera ?: return true
                val z = cam.cameraInfo.zoomState.value ?: return true
                val ratio = (z.zoomRatio * d.scaleFactor).coerceIn(z.minZoomRatio, z.maxZoomRatio)
                cam.cameraControl.setZoomRatio(ratio)
                return true
            }
        })
        preview.setOnTouchListener { _, ev -> pinch.onTouchEvent(ev); true }
    }

    private fun loadPicked(uri: Uri) {
        var name = "своя модель"
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) name = it.getString(0)
        }
        val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return
        File(filesDir, "custom.onnx").writeBytes(bytes)
        getPreferences(MODE_PRIVATE).edit().putString("model", name).apply()
        loadModel(bytes, name)
    }

    /** New session on the analysis thread, so it never swaps under a running inference. */
    private fun loadModel(bytes: ByteArray, name: String) {
        modelBytes = bytes
        exec.execute {
            try {
                val opts = OrtSession.SessionOptions().apply {
                    when (engine) {
                        "XNNPACK" -> { setIntraOpNumThreads(1); addXnnpack(mapOf("intra_op_num_threads" to "4")) }
                        "NNAPI" -> addNnapi()
                        else -> setIntraOpNumThreads(4)
                    }
                }
                val ns = env.createSession(bytes, opts)
                val outShape = (ns.outputInfo.values.first().info as ai.onnxruntime.TensorInfo).shape
                require(outShape.size == 3 && outShape[2] == 6L) {
                    "вихід моделі ${outShape.contentToString()}, очікувався [1, N, 6] (YOLO26 end-to-end)"
                }
                val old = session
                session = ns
                old?.close()
                modelName = name
            } catch (e: Exception) {
                runOnUiThread { info.text = "Модель не завантажилась: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(1280, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val pv = Preview.Builder().setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY).build()
            ).build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(exec) { analyze(it) }
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, pv, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(img: ImageProxy) {
        val s = session
        if (s == null) { img.close(); return }
        val rot = img.imageInfo.rotationDegrees
        val raw = img.toBitmap()
        img.close()
        val bmp = if (rot != 0) Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height,
            Matrix().apply { postRotate(rot.toFloat()) }, true) else raw
        val w = bmp.width
        val h = bmp.height

        // letterbox into 640x640, grey 114 - exactly the station's / Ultralytics' preprocessing
        val r = min(size / w.toFloat(), size / h.toFloat())
        val nw = (w * r).roundToInt()
        val nh = (h * r).roundToInt()
        val px = (size - nw) / 2
        val py = (size - nh) / 2
        val lb = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(lb).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(Bitmap.createScaledBitmap(bmp, nw, nh, true), px.toFloat(), py.toFloat(), null)
        }
        val pixels = IntArray(size * size)
        lb.getPixels(pixels, 0, size, 0, 0, size, size)
        val plane = size * size
        val buf = FloatBuffer.allocate(3 * plane)
        for (i in 0 until plane) {
            val p = pixels[i]
            buf.put(i, ((p shr 16) and 255) / 255f)
            buf.put(plane + i, ((p shr 8) and 255) / 255f)
            buf.put(2 * plane + i, (p and 255) / 255f)
        }

        val t0 = SystemClock.elapsedRealtime()
        val dets = ArrayList<Det>()
        OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, size.toLong(), size.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { res ->
                @Suppress("UNCHECKED_CAST")
                val out = (res[0].value as Array<Array<FloatArray>>)[0]
                for (row in out) {
                    if (row[4] < thr) continue
                    val x1 = ((row[0] - px) / r).coerceIn(0f, w.toFloat())
                    val y1 = ((row[1] - py) / r).coerceIn(0f, h.toFloat())
                    val x2 = ((row[2] - px) / r).coerceIn(0f, w.toFloat())
                    val y2 = ((row[3] - py) / r).coerceIn(0f, h.toFloat())
                    if (x2 > x1 && y2 > y1) dets.add(Det(RectF(x1, y1, x2, y2), row[4]))
                }
            }
        }
        val now = SystemClock.elapsedRealtime()
        inferMs = if (inferMs == 0f) (now - t0).toFloat() else 0.9f * inferMs + 0.1f * (now - t0)
        if (lastT != 0L) {
            val f = 1000f / (now - lastT).coerceAtLeast(1)
            fps = if (fps == 0f) f else 0.9f * fps + 0.1f * f
        }
        lastT = now
        overlay.show(dets, w, h)
        val zoom = camera?.cameraInfo?.zoomState?.value?.zoomRatio ?: 1f
        runOnUiThread {
            info.text = "Модель: $modelName · $engine (торкнись — змінити)\n%.1f кадр/с · модель %.0f мс · zoom %.1fx · рамок: %d"
                .format(fps, inferMs, zoom, dets.size)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        exec.shutdown()
        session?.close()
    }
}
