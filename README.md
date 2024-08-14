# Vehicle counter

## Import Library


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
```

### import cv2
Library: OpenCV (Open Source Computer Vision Library).

Fungsi: Digunakan untuk pemrosesan gambar dan video. Ini termasuk membaca, menulis, dan memanipulasi gambar dan video, serta melakukan berbagai tugas analisis gambar seperti deteksi objek, pelacakan, dan segmentasi.

### import numpy as np
Library: NumPy (Numerical Python).

Fungsi: Digunakan untuk komputasi numerik dan operasi array. NumPy menyediakan struktur data array multidimensi (ndarray) dan fungsi-fungsi matematika yang efisien.

### import matplotlib.pyplot as plt
Library: Matplotlib.

Fungsi: Digunakan untuk visualisasi data. matplotlib.pyplot adalah modul dari Matplotlib yang menyediakan fungsi-fungsi untuk membuat berbagai jenis plot, seperti grafik garis, histogram, dan scatter plots.

### from collections import defaultdict
Library: Collections (bagian dari pustaka standar Python).

Fungsi: defaultdict adalah subclass dari dictionary yang memungkinkan Anda menentukan nilai default untuk key yang tidak ada. Ini berguna untuk menghindari pengecekan apakah key ada sebelum mengakses atau menambah nilai.

## PATH


```python
weights_path = "yolov4.weights"  
cfg_path = "yolov4.cfg"          
names_path = "coco.names"
video_path = "video/toll_gate.mp4"
points = [(8, 160), (70, 175), (135, 190), (205, 205), (279, 221), (340, 235), (422, 255), (480, 268), (561, 285)]
```

### Path 
Berisi path - path yang diperlukan seperti yolo dan video, untuk points adalah koordinat dari line gate yang akan dibuat

## Load Video


```python
def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return None
    return video
  
video = load_video(video_path)

if video is not None:
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}")

if video is not None:
    ret, frame = video.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off') 
        plt.show()
        
if video is not None:
    video.release()
```

    Total number of frames: 358
    


    
![png](output_8_1.png)
    


1. Memuat Video: Menggunakan OpenCV untuk membuka dan membaca video.
2. Menampilkan Jumlah Frame: Menghitung dan menampilkan total jumlah frame dalam video.
3. Menampilkan Frame Pertama: Membaca dan menampilkan frame pertama dari video menggunakan Matplotlib.
4. Menutup Video: Menutup file video untuk membebaskan sumber daya.

## Load YOLO (You Only Look Once)


```python
def load_yolo_model(weights_path=weights_path, cfg_path=cfg_path, names_path=names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers, classes
```

Fungsi load_yolo_model memuat model YOLO untuk deteksi objek dengan:

1. Mengambil model YOLO dari file konfigurasi dan bobot.
2. Membaca nama kelas dari file nama kelas.
3. Mengidentifikasi nama layer output dari model.
4. Mengembalikan objek model, layer output, dan daftar nama kelas yang dikenali oleh model.

## Membuat Gate Lines


```python
def draw_gate_lines(frame):  
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 3)
        
        mid_x = (points[i][0] + points[i + 1][0]) // 2
        mid_y = (points[i][1] + points[i + 1][1]) // 2
        
        cv2.putText(frame, str(i + 1), (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    return frame
```

1. Gambar Garis: Menggunakan cv2.line untuk menggambar garis antara pasangan titik yang berurutan dalam daftar points.
2. Tambahkan Label: Menggunakan cv2.putText untuk menambahkan label di tengah setiap garis.
3. Kembalikan Frame: Mengembalikan frame yang telah dimodifikasi untuk digunakan lebih lanjut, seperti menampilkan atau menyimpan gambar.

## Deteksi Object


```python
def detect_objects(frame, net, output_layers, target_classes=["car", "bus"]):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]  
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = str(classes[class_id])
            
            if confidence > 0.5 and label in target_classes:  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], class_ids[i], confidences[i]) for i in indices]

video = cv2.VideoCapture(video_path)

net, output_layers, classes = load_yolo_model(weights_path, cfg_path, names_path)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    detections = detect_objects(frame, net, output_layers)
    
    for box, class_id, confidence in detections:
        label = str(classes[class_id])
        x, y, w, h = box
        color = (0, 0, 255) if label == "car" else (0, 255, 0) if label == "bus" else (255, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    frame = draw_gate_lines(frame)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

video.release()
```


    
![png](output_17_0.png)
    



    
![png](output_17_1.png)
    



    
![png](output_17_2.png)
    



    
![png](output_17_3.png)
    



    
![png](output_17_4.png)
    



    
![png](output_17_5.png)
    



    
![png](output_17_6.png)
    



    
![png](output_17_7.png)
    



    
![png](output_17_8.png)
    



    
![png](output_17_9.png)
    



    
![png](output_17_10.png)
    



    
![png](output_17_11.png)
    



    
![png](output_17_12.png)
    



    
![png](output_17_13.png)
    



    
![png](output_17_14.png)
    



    
![png](output_17_15.png)
    



    
![png](output_17_16.png)
    



    
![png](output_17_17.png)
    



    
![png](output_17_18.png)
    



    
![png](output_17_19.png)
    



    
![png](output_17_20.png)
    



    
![png](output_17_21.png)
    



    
![png](output_17_22.png)
    



    
![png](output_17_23.png)
    



    
![png](output_17_24.png)
    



    
![png](output_17_25.png)
    



    
![png](output_17_26.png)
    



    
![png](output_17_27.png)
    



    
![png](output_17_28.png)
    



    
![png](output_17_29.png)
    



    
![png](output_17_30.png)
    



    
![png](output_17_31.png)
    



    
![png](output_17_32.png)
    



    
![png](output_17_33.png)
    



    
![png](output_17_34.png)
    



    
![png](output_17_35.png)
    



    
![png](output_17_36.png)
    



    
![png](output_17_37.png)
    



    
![png](output_17_38.png)
    



    
![png](output_17_39.png)
    



    
![png](output_17_40.png)
    



    
![png](output_17_41.png)
    



    
![png](output_17_42.png)
    



    
![png](output_17_43.png)
    



    
![png](output_17_44.png)
    



    
![png](output_17_45.png)
    



    
![png](output_17_46.png)
    



    
![png](output_17_47.png)
    



    
![png](output_17_48.png)
    



    
![png](output_17_49.png)
    



    
![png](output_17_50.png)
    



    
![png](output_17_51.png)
    



    
![png](output_17_52.png)
    



    
![png](output_17_53.png)
    



    
![png](output_17_54.png)
    



    
![png](output_17_55.png)
    



    
![png](output_17_56.png)
    



    
![png](output_17_57.png)
    



    
![png](output_17_58.png)
    



    
![png](output_17_59.png)
    



    
![png](output_17_60.png)
    



    
![png](output_17_61.png)
    



    
![png](output_17_62.png)
    



    
![png](output_17_63.png)
    



    
![png](output_17_64.png)
    



    
![png](output_17_65.png)
    



    
![png](output_17_66.png)
    



    
![png](output_17_67.png)
    



    
![png](output_17_68.png)
    



    
![png](output_17_69.png)
    



    
![png](output_17_70.png)
    



    
![png](output_17_71.png)
    



    
![png](output_17_72.png)
    



    
![png](output_17_73.png)
    



    
![png](output_17_74.png)
    



    
![png](output_17_75.png)
    



    
![png](output_17_76.png)
    



    
![png](output_17_77.png)
    



    
![png](output_17_78.png)
    



    
![png](output_17_79.png)
    



    
![png](output_17_80.png)
    



    
![png](output_17_81.png)
    



    
![png](output_17_82.png)
    



    
![png](output_17_83.png)
    



    
![png](output_17_84.png)
    



    
![png](output_17_85.png)
    



    
![png](output_17_86.png)
    



    
![png](output_17_87.png)
    



    
![png](output_17_88.png)
    



    
![png](output_17_89.png)
    



    
![png](output_17_90.png)
    



    
![png](output_17_91.png)
    



    
![png](output_17_92.png)
    



    
![png](output_17_93.png)
    



    
![png](output_17_94.png)
    



    
![png](output_17_95.png)
    



    
![png](output_17_96.png)
    



    
![png](output_17_97.png)
    



    
![png](output_17_98.png)
    



    
![png](output_17_99.png)
    



    
![png](output_17_100.png)
    



    
![png](output_17_101.png)
    



    
![png](output_17_102.png)
    



    
![png](output_17_103.png)
    



    
![png](output_17_104.png)
    



    
![png](output_17_105.png)
    



    
![png](output_17_106.png)
    



    
![png](output_17_107.png)
    



    
![png](output_17_108.png)
    



    
![png](output_17_109.png)
    



    
![png](output_17_110.png)
    



    
![png](output_17_111.png)
    



    
![png](output_17_112.png)
    



    
![png](output_17_113.png)
    



    
![png](output_17_114.png)
    



    
![png](output_17_115.png)
    



    
![png](output_17_116.png)
    



    
![png](output_17_117.png)
    



    
![png](output_17_118.png)
    



    
![png](output_17_119.png)
    



    
![png](output_17_120.png)
    



    
![png](output_17_121.png)
    



    
![png](output_17_122.png)
    



    
![png](output_17_123.png)
    



    
![png](output_17_124.png)
    



    
![png](output_17_125.png)
    



    
![png](output_17_126.png)
    



    
![png](output_17_127.png)
    



    
![png](output_17_128.png)
    



    
![png](output_17_129.png)
    



    
![png](output_17_130.png)
    



    
![png](output_17_131.png)
    



    
![png](output_17_132.png)
    



    
![png](output_17_133.png)
    



    
![png](output_17_134.png)
    



    
![png](output_17_135.png)
    



    
![png](output_17_136.png)
    



    
![png](output_17_137.png)
    



    
![png](output_17_138.png)
    



    
![png](output_17_139.png)
    



    
![png](output_17_140.png)
    



    
![png](output_17_141.png)
    



    
![png](output_17_142.png)
    



    
![png](output_17_143.png)
    



    
![png](output_17_144.png)
    



    
![png](output_17_145.png)
    



    
![png](output_17_146.png)
    



    
![png](output_17_147.png)
    



    
![png](output_17_148.png)
    



    
![png](output_17_149.png)
    



    
![png](output_17_150.png)
    



    
![png](output_17_151.png)
    



    
![png](output_17_152.png)
    



    
![png](output_17_153.png)
    



    
![png](output_17_154.png)
    



    
![png](output_17_155.png)
    



    
![png](output_17_156.png)
    



    
![png](output_17_157.png)
    



    
![png](output_17_158.png)
    



    
![png](output_17_159.png)
    



    
![png](output_17_160.png)
    



    
![png](output_17_161.png)
    



    
![png](output_17_162.png)
    



    
![png](output_17_163.png)
    



    
![png](output_17_164.png)
    



    
![png](output_17_165.png)
    



    
![png](output_17_166.png)
    



    
![png](output_17_167.png)
    



    
![png](output_17_168.png)
    



    
![png](output_17_169.png)
    



    
![png](output_17_170.png)
    



    
![png](output_17_171.png)
    



    
![png](output_17_172.png)
    



    
![png](output_17_173.png)
    



    
![png](output_17_174.png)
    



    
![png](output_17_175.png)
    



    
![png](output_17_176.png)
    



    
![png](output_17_177.png)
    



    
![png](output_17_178.png)
    



    
![png](output_17_179.png)
    



    
![png](output_17_180.png)
    



    
![png](output_17_181.png)
    



    
![png](output_17_182.png)
    



    
![png](output_17_183.png)
    



    
![png](output_17_184.png)
    



    
![png](output_17_185.png)
    



    
![png](output_17_186.png)
    



    
![png](output_17_187.png)
    



    
![png](output_17_188.png)
    



    
![png](output_17_189.png)
    



    
![png](output_17_190.png)
    



    
![png](output_17_191.png)
    



    
![png](output_17_192.png)
    



    
![png](output_17_193.png)
    



    
![png](output_17_194.png)
    



    
![png](output_17_195.png)
    



    
![png](output_17_196.png)
    



    
![png](output_17_197.png)
    



    
![png](output_17_198.png)
    



    
![png](output_17_199.png)
    



    
![png](output_17_200.png)
    



    
![png](output_17_201.png)
    



    
![png](output_17_202.png)
    



    
![png](output_17_203.png)
    



    
![png](output_17_204.png)
    



    
![png](output_17_205.png)
    



    
![png](output_17_206.png)
    



    
![png](output_17_207.png)
    



    
![png](output_17_208.png)
    



    
![png](output_17_209.png)
    



    
![png](output_17_210.png)
    



    
![png](output_17_211.png)
    



    
![png](output_17_212.png)
    



    
![png](output_17_213.png)
    



    
![png](output_17_214.png)
    



    
![png](output_17_215.png)
    



    
![png](output_17_216.png)
    



    
![png](output_17_217.png)
    



    
![png](output_17_218.png)
    



    
![png](output_17_219.png)
    



    
![png](output_17_220.png)
    



    
![png](output_17_221.png)
    



    
![png](output_17_222.png)
    



    
![png](output_17_223.png)
    



    
![png](output_17_224.png)
    



    
![png](output_17_225.png)
    



    
![png](output_17_226.png)
    



    
![png](output_17_227.png)
    



    
![png](output_17_228.png)
    



    
![png](output_17_229.png)
    



    
![png](output_17_230.png)
    



    
![png](output_17_231.png)
    



    
![png](output_17_232.png)
    



    
![png](output_17_233.png)
    



    
![png](output_17_234.png)
    



    
![png](output_17_235.png)
    



    
![png](output_17_236.png)
    



    
![png](output_17_237.png)
    



    
![png](output_17_238.png)
    



    
![png](output_17_239.png)
    



    
![png](output_17_240.png)
    



    
![png](output_17_241.png)
    



    
![png](output_17_242.png)
    



    
![png](output_17_243.png)
    



    
![png](output_17_244.png)
    



    
![png](output_17_245.png)
    



    
![png](output_17_246.png)
    



    
![png](output_17_247.png)
    



    
![png](output_17_248.png)
    



    
![png](output_17_249.png)
    



    
![png](output_17_250.png)
    



    
![png](output_17_251.png)
    



    
![png](output_17_252.png)
    



    
![png](output_17_253.png)
    



    
![png](output_17_254.png)
    



    
![png](output_17_255.png)
    



    
![png](output_17_256.png)
    



    
![png](output_17_257.png)
    



    
![png](output_17_258.png)
    



    
![png](output_17_259.png)
    



    
![png](output_17_260.png)
    



    
![png](output_17_261.png)
    



    
![png](output_17_262.png)
    



    
![png](output_17_263.png)
    



    
![png](output_17_264.png)
    



    
![png](output_17_265.png)
    



    
![png](output_17_266.png)
    



    
![png](output_17_267.png)
    



    
![png](output_17_268.png)
    



    
![png](output_17_269.png)
    



    
![png](output_17_270.png)
    



    
![png](output_17_271.png)
    



    
![png](output_17_272.png)
    



    
![png](output_17_273.png)
    



    
![png](output_17_274.png)
    



    
![png](output_17_275.png)
    



    
![png](output_17_276.png)
    



    
![png](output_17_277.png)
    



    
![png](output_17_278.png)
    



    
![png](output_17_279.png)
    



    
![png](output_17_280.png)
    



    
![png](output_17_281.png)
    



    
![png](output_17_282.png)
    



    
![png](output_17_283.png)
    



    
![png](output_17_284.png)
    



    
![png](output_17_285.png)
    



    
![png](output_17_286.png)
    



    
![png](output_17_287.png)
    



    
![png](output_17_288.png)
    



    
![png](output_17_289.png)
    



    
![png](output_17_290.png)
    



    
![png](output_17_291.png)
    



    
![png](output_17_292.png)
    



    
![png](output_17_293.png)
    



    
![png](output_17_294.png)
    



    
![png](output_17_295.png)
    



    
![png](output_17_296.png)
    



    
![png](output_17_297.png)
    



    
![png](output_17_298.png)
    



    
![png](output_17_299.png)
    



    
![png](output_17_300.png)
    



    
![png](output_17_301.png)
    



    
![png](output_17_302.png)
    



    
![png](output_17_303.png)
    



    
![png](output_17_304.png)
    



    
![png](output_17_305.png)
    



    
![png](output_17_306.png)
    



    
![png](output_17_307.png)
    



    
![png](output_17_308.png)
    



    
![png](output_17_309.png)
    



    
![png](output_17_310.png)
    



    
![png](output_17_311.png)
    



    
![png](output_17_312.png)
    



    
![png](output_17_313.png)
    



    
![png](output_17_314.png)
    



    
![png](output_17_315.png)
    



    
![png](output_17_316.png)
    



    
![png](output_17_317.png)
    



    
![png](output_17_318.png)
    



    
![png](output_17_319.png)
    



    
![png](output_17_320.png)
    



    
![png](output_17_321.png)
    



    
![png](output_17_322.png)
    



    
![png](output_17_323.png)
    



    
![png](output_17_324.png)
    



    
![png](output_17_325.png)
    



    
![png](output_17_326.png)
    



    
![png](output_17_327.png)
    



    
![png](output_17_328.png)
    



    
![png](output_17_329.png)
    



    
![png](output_17_330.png)
    



    
![png](output_17_331.png)
    



    
![png](output_17_332.png)
    



    
![png](output_17_333.png)
    



    
![png](output_17_334.png)
    



    
![png](output_17_335.png)
    



    
![png](output_17_336.png)
    



    
![png](output_17_337.png)
    



    
![png](output_17_338.png)
    



    
![png](output_17_339.png)
    



    
![png](output_17_340.png)
    



    
![png](output_17_341.png)
    



    
![png](output_17_342.png)
    



    
![png](output_17_343.png)
    



    
![png](output_17_344.png)
    



    
![png](output_17_345.png)
    



    
![png](output_17_346.png)
    



    
![png](output_17_347.png)
    



    
![png](output_17_348.png)
    



    
![png](output_17_349.png)
    



    
![png](output_17_350.png)
    



    
![png](output_17_351.png)
    



    
![png](output_17_352.png)
    



    
![png](output_17_353.png)
    



    
![png](output_17_354.png)
    



    
![png](output_17_355.png)
    



    
![png](output_17_356.png)
    



    
![png](output_17_357.png)
    


1. Memuat dan Mengonfigurasi Model YOLO: Memuat model YOLO dari file bobot, konfigurasi, dan nama kelas.
2. Mendeteksi Objek dalam Setiap Frame: Menggunakan model untuk mendeteksi objek dalam setiap frame video.
3. Visualisasi Deteksi: Menggambar kotak pembatas dan label untuk deteksi objek dalam frame.
4. Menampilkan dan Menggambar Garis pada Frame: Menampilkan frame yang telah dimodifikasi dan menggambar garis dengan fungsi draw_gate_lines.
5. Menutup Video: Melepaskan file video setelah selesai memproses.

## Tracking


```python
class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.nextObjectID = 1
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int((x + x + w) / 2.0)
            cY = int((y + y + h) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids) - input_centroids[:, np.newaxis], axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[col]
                self.objects[objectID] = input_centroids[row]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    self.register(input_centroids[row])
            else:
                for col in unusedCols:
                    objectID = objectIDs[col]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)

        return self.objects
```

Kelas CentroidTracker digunakan untuk melacak objek yang bergerak dalam video berdasarkan posisi sentroid (titik tengah) dari bounding box objek. Fungsi utamanya adalah:

1. register(): Mendaftarkan objek baru yang terdeteksi.
2. deregister(): Menghapus objek yang telah menghilang dari pelacakan.
3. update(): Memperbarui posisi objek berdasarkan sentroid dari deteksi terbaru, mencocokkan objek yang sudah ada dengan deteksi baru, dan menghitung objek yang menghilang atau muncul kembali.

## Count dan Visualisasi


```python
def has_crossed_line(centroid, line_y):
    return centroid[1] > line_y

video = cv2.VideoCapture(video_path)

tracker = CentroidTracker()  
car_count = 0
bus_count = 0

gate_y = points[-1][1]  

counted_objects = set()

while video.isOpened():
    ret, frame = video.read()
    if not ret or frame is None:
        break  

    detections = detect_objects(frame, net, output_layers)
    
    rects = [box for (box, class_id, confidence) in detections if classes[class_id] in ["car", "bus"]]
    
    objects = tracker.update(rects)
    
    for objectID, centroid in objects.items():
        vehicle_class = None
        for box, class_id, confidence in detections:
            if classes[class_id] in ["car", "bus"]:
                x, y, w, h = box
                if (x < centroid[0] < x + w) and (y < centroid[1] < y + h):
                    vehicle_class = classes[class_id]
                    break
        
        if has_crossed_line(centroid, gate_y) and objectID not in counted_objects:
            if vehicle_class == "car":
                car_count += 1
            elif vehicle_class == "bus":
                bus_count += 1
            counted_objects.add(objectID)
            
    # VISUALISASI
    for objectID, centroid in objects.items():
        vehicle_class = None
        for box, class_id, confidence in detections:
            if classes[class_id] in ["car", "bus"]:
                x, y, w, h = box
                if (x < centroid[0] < x + w) and (y < centroid[1] < y + h):
                    vehicle_class = classes[class_id]
                    break

        color = (0, 0, 255) if vehicle_class == "car" else (0, 255, 0) if vehicle_class == "bus" else (255, 255, 255)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if vehicle_class:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{vehicle_class} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame = draw_gate_lines(frame)
    frame_width = frame.shape[1]
    cv2.putText(frame, "Created by Minan A", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f"Mobil: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Bus  : {bus_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

video.release()
```


    
![png](output_23_0.png)
    



    
![png](output_23_1.png)
    



    
![png](output_23_2.png)
    



    
![png](output_23_3.png)
    



    
![png](output_23_4.png)
    



    
![png](output_23_5.png)
    



    
![png](output_23_6.png)
    



    
![png](output_23_7.png)
    



    
![png](output_23_8.png)
    



    
![png](output_23_9.png)
    



    
![png](output_23_10.png)
    



    
![png](output_23_11.png)
    



    
![png](output_23_12.png)
    



    
![png](output_23_13.png)
    



    
![png](output_23_14.png)
    



    
![png](output_23_15.png)
    



    
![png](output_23_16.png)
    



    
![png](output_23_17.png)
    



    
![png](output_23_18.png)
    



    
![png](output_23_19.png)
    



    
![png](output_23_20.png)
    



    
![png](output_23_21.png)
    



    
![png](output_23_22.png)
    



    
![png](output_23_23.png)
    



    
![png](output_23_24.png)
    



    
![png](output_23_25.png)
    



    
![png](output_23_26.png)
    



    
![png](output_23_27.png)
    



    
![png](output_23_28.png)
    



    
![png](output_23_29.png)
    



    
![png](output_23_30.png)
    



    
![png](output_23_31.png)
    



    
![png](output_23_32.png)
    



    
![png](output_23_33.png)
    



    
![png](output_23_34.png)
    



    
![png](output_23_35.png)
    



    
![png](output_23_36.png)
    



    
![png](output_23_37.png)
    



    
![png](output_23_38.png)
    



    
![png](output_23_39.png)
    



    
![png](output_23_40.png)
    



    
![png](output_23_41.png)
    



    
![png](output_23_42.png)
    



    
![png](output_23_43.png)
    



    
![png](output_23_44.png)
    



    
![png](output_23_45.png)
    



    
![png](output_23_46.png)
    



    
![png](output_23_47.png)
    



    
![png](output_23_48.png)
    



    
![png](output_23_49.png)
    



    
![png](output_23_50.png)
    



    
![png](output_23_51.png)
    



    
![png](output_23_52.png)
    



    
![png](output_23_53.png)
    



    
![png](output_23_54.png)
    



    
![png](output_23_55.png)
    



    
![png](output_23_56.png)
    



    
![png](output_23_57.png)
    



    
![png](output_23_58.png)
    



    
![png](output_23_59.png)
    



    
![png](output_23_60.png)
    



    
![png](output_23_61.png)
    



    
![png](output_23_62.png)
    



    
![png](output_23_63.png)
    



    
![png](output_23_64.png)
    



    
![png](output_23_65.png)
    



    
![png](output_23_66.png)
    



    
![png](output_23_67.png)
    



    
![png](output_23_68.png)
    



    
![png](output_23_69.png)
    



    
![png](output_23_70.png)
    



    
![png](output_23_71.png)
    



    
![png](output_23_72.png)
    



    
![png](output_23_73.png)
    



    
![png](output_23_74.png)
    



    
![png](output_23_75.png)
    



    
![png](output_23_76.png)
    



    
![png](output_23_77.png)
    



    
![png](output_23_78.png)
    



    
![png](output_23_79.png)
    



    
![png](output_23_80.png)
    



    
![png](output_23_81.png)
    



    
![png](output_23_82.png)
    



    
![png](output_23_83.png)
    



    
![png](output_23_84.png)
    



    
![png](output_23_85.png)
    



    
![png](output_23_86.png)
    



    
![png](output_23_87.png)
    



    
![png](output_23_88.png)
    



    
![png](output_23_89.png)
    



    
![png](output_23_90.png)
    



    
![png](output_23_91.png)
    



    
![png](output_23_92.png)
    



    
![png](output_23_93.png)
    



    
![png](output_23_94.png)
    



    
![png](output_23_95.png)
    



    
![png](output_23_96.png)
    



    
![png](output_23_97.png)
    



    
![png](output_23_98.png)
    



    
![png](output_23_99.png)
    



    
![png](output_23_100.png)
    



    
![png](output_23_101.png)
    



    
![png](output_23_102.png)
    



    
![png](output_23_103.png)
    



    
![png](output_23_104.png)
    



    
![png](output_23_105.png)
    



    
![png](output_23_106.png)
    



    
![png](output_23_107.png)
    



    
![png](output_23_108.png)
    



    
![png](output_23_109.png)
    



    
![png](output_23_110.png)
    



    
![png](output_23_111.png)
    



    
![png](output_23_112.png)
    



    
![png](output_23_113.png)
    



    
![png](output_23_114.png)
    



    
![png](output_23_115.png)
    



    
![png](output_23_116.png)
    



    
![png](output_23_117.png)
    



    
![png](output_23_118.png)
    



    
![png](output_23_119.png)
    



    
![png](output_23_120.png)
    



    
![png](output_23_121.png)
    



    
![png](output_23_122.png)
    



    
![png](output_23_123.png)
    



    
![png](output_23_124.png)
    



    
![png](output_23_125.png)
    



    
![png](output_23_126.png)
    



    
![png](output_23_127.png)
    



    
![png](output_23_128.png)
    



    
![png](output_23_129.png)
    



    
![png](output_23_130.png)
    



    
![png](output_23_131.png)
    



    
![png](output_23_132.png)
    



    
![png](output_23_133.png)
    



    
![png](output_23_134.png)
    



    
![png](output_23_135.png)
    



    
![png](output_23_136.png)
    



    
![png](output_23_137.png)
    



    
![png](output_23_138.png)
    



    
![png](output_23_139.png)
    



    
![png](output_23_140.png)
    



    
![png](output_23_141.png)
    



    
![png](output_23_142.png)
    



    
![png](output_23_143.png)
    



    
![png](output_23_144.png)
    



    
![png](output_23_145.png)
    



    
![png](output_23_146.png)
    



    
![png](output_23_147.png)
    



    
![png](output_23_148.png)
    



    
![png](output_23_149.png)
    



    
![png](output_23_150.png)
    



    
![png](output_23_151.png)
    



    
![png](output_23_152.png)
    



    
![png](output_23_153.png)
    



    
![png](output_23_154.png)
    



    
![png](output_23_155.png)
    



    
![png](output_23_156.png)
    



    
![png](output_23_157.png)
    



    
![png](output_23_158.png)
    



    
![png](output_23_159.png)
    



    
![png](output_23_160.png)
    



    
![png](output_23_161.png)
    



    
![png](output_23_162.png)
    



    
![png](output_23_163.png)
    



    
![png](output_23_164.png)
    



    
![png](output_23_165.png)
    



    
![png](output_23_166.png)
    



    
![png](output_23_167.png)
    



    
![png](output_23_168.png)
    



    
![png](output_23_169.png)
    



    
![png](output_23_170.png)
    



    
![png](output_23_171.png)
    



    
![png](output_23_172.png)
    



    
![png](output_23_173.png)
    



    
![png](output_23_174.png)
    



    
![png](output_23_175.png)
    



    
![png](output_23_176.png)
    



    
![png](output_23_177.png)
    



    
![png](output_23_178.png)
    



    
![png](output_23_179.png)
    



    
![png](output_23_180.png)
    



    
![png](output_23_181.png)
    



    
![png](output_23_182.png)
    



    
![png](output_23_183.png)
    



    
![png](output_23_184.png)
    



    
![png](output_23_185.png)
    



    
![png](output_23_186.png)
    



    
![png](output_23_187.png)
    



    
![png](output_23_188.png)
    



    
![png](output_23_189.png)
    



    
![png](output_23_190.png)
    



    
![png](output_23_191.png)
    



    
![png](output_23_192.png)
    



    
![png](output_23_193.png)
    



    
![png](output_23_194.png)
    



    
![png](output_23_195.png)
    



    
![png](output_23_196.png)
    



    
![png](output_23_197.png)
    



    
![png](output_23_198.png)
    



    
![png](output_23_199.png)
    



    
![png](output_23_200.png)
    



    
![png](output_23_201.png)
    



    
![png](output_23_202.png)
    



    
![png](output_23_203.png)
    



    
![png](output_23_204.png)
    



    
![png](output_23_205.png)
    



    
![png](output_23_206.png)
    



    
![png](output_23_207.png)
    



    
![png](output_23_208.png)
    



    
![png](output_23_209.png)
    



    
![png](output_23_210.png)
    



    
![png](output_23_211.png)
    



    
![png](output_23_212.png)
    



    
![png](output_23_213.png)
    



    
![png](output_23_214.png)
    



    
![png](output_23_215.png)
    



    
![png](output_23_216.png)
    



    
![png](output_23_217.png)
    



    
![png](output_23_218.png)
    



    
![png](output_23_219.png)
    



    
![png](output_23_220.png)
    



    
![png](output_23_221.png)
    



    
![png](output_23_222.png)
    



    
![png](output_23_223.png)
    



    
![png](output_23_224.png)
    



    
![png](output_23_225.png)
    



    
![png](output_23_226.png)
    



    
![png](output_23_227.png)
    



    
![png](output_23_228.png)
    



    
![png](output_23_229.png)
    



    
![png](output_23_230.png)
    



    
![png](output_23_231.png)
    



    
![png](output_23_232.png)
    



    
![png](output_23_233.png)
    



    
![png](output_23_234.png)
    



    
![png](output_23_235.png)
    



    
![png](output_23_236.png)
    



    
![png](output_23_237.png)
    



    
![png](output_23_238.png)
    



    
![png](output_23_239.png)
    



    
![png](output_23_240.png)
    



    
![png](output_23_241.png)
    



    
![png](output_23_242.png)
    



    
![png](output_23_243.png)
    



    
![png](output_23_244.png)
    



    
![png](output_23_245.png)
    



    
![png](output_23_246.png)
    



    
![png](output_23_247.png)
    



    
![png](output_23_248.png)
    



    
![png](output_23_249.png)
    



    
![png](output_23_250.png)
    



    
![png](output_23_251.png)
    



    
![png](output_23_252.png)
    



    
![png](output_23_253.png)
    



    
![png](output_23_254.png)
    



    
![png](output_23_255.png)
    



    
![png](output_23_256.png)
    



    
![png](output_23_257.png)
    



    
![png](output_23_258.png)
    



    
![png](output_23_259.png)
    



    
![png](output_23_260.png)
    



    
![png](output_23_261.png)
    



    
![png](output_23_262.png)
    



    
![png](output_23_263.png)
    



    
![png](output_23_264.png)
    



    
![png](output_23_265.png)
    



    
![png](output_23_266.png)
    



    
![png](output_23_267.png)
    



    
![png](output_23_268.png)
    



    
![png](output_23_269.png)
    



    
![png](output_23_270.png)
    



    
![png](output_23_271.png)
    



    
![png](output_23_272.png)
    



    
![png](output_23_273.png)
    



    
![png](output_23_274.png)
    



    
![png](output_23_275.png)
    



    
![png](output_23_276.png)
    



    
![png](output_23_277.png)
    



    
![png](output_23_278.png)
    



    
![png](output_23_279.png)
    



    
![png](output_23_280.png)
    



    
![png](output_23_281.png)
    



    
![png](output_23_282.png)
    



    
![png](output_23_283.png)
    



    
![png](output_23_284.png)
    



    
![png](output_23_285.png)
    



    
![png](output_23_286.png)
    



    
![png](output_23_287.png)
    



    
![png](output_23_288.png)
    



    
![png](output_23_289.png)
    



    
![png](output_23_290.png)
    



    
![png](output_23_291.png)
    



    
![png](output_23_292.png)
    



    
![png](output_23_293.png)
    



    
![png](output_23_294.png)
    



    
![png](output_23_295.png)
    



    
![png](output_23_296.png)
    



    
![png](output_23_297.png)
    



    
![png](output_23_298.png)
    



    
![png](output_23_299.png)
    



    
![png](output_23_300.png)
    



    
![png](output_23_301.png)
    



    
![png](output_23_302.png)
    



    
![png](output_23_303.png)
    



    
![png](output_23_304.png)
    



    
![png](output_23_305.png)
    



    
![png](output_23_306.png)
    



    
![png](output_23_307.png)
    



    
![png](output_23_308.png)
    



    
![png](output_23_309.png)
    



    
![png](output_23_310.png)
    



    
![png](output_23_311.png)
    



    
![png](output_23_312.png)
    



    
![png](output_23_313.png)
    



    
![png](output_23_314.png)
    



    
![png](output_23_315.png)
    



    
![png](output_23_316.png)
    



    
![png](output_23_317.png)
    



    
![png](output_23_318.png)
    



    
![png](output_23_319.png)
    



    
![png](output_23_320.png)
    



    
![png](output_23_321.png)
    



    
![png](output_23_322.png)
    



    
![png](output_23_323.png)
    



    
![png](output_23_324.png)
    



    
![png](output_23_325.png)
    



    
![png](output_23_326.png)
    



    
![png](output_23_327.png)
    



    
![png](output_23_328.png)
    



    
![png](output_23_329.png)
    



    
![png](output_23_330.png)
    



    
![png](output_23_331.png)
    



    
![png](output_23_332.png)
    



    
![png](output_23_333.png)
    



    
![png](output_23_334.png)
    



    
![png](output_23_335.png)
    



    
![png](output_23_336.png)
    



    
![png](output_23_337.png)
    



    
![png](output_23_338.png)
    



    
![png](output_23_339.png)
    



    
![png](output_23_340.png)
    



    
![png](output_23_341.png)
    



    
![png](output_23_342.png)
    



    
![png](output_23_343.png)
    



    
![png](output_23_344.png)
    



    
![png](output_23_345.png)
    



    
![png](output_23_346.png)
    



    
![png](output_23_347.png)
    



    
![png](output_23_348.png)
    



    
![png](output_23_349.png)
    



    
![png](output_23_350.png)
    



    
![png](output_23_351.png)
    



    
![png](output_23_352.png)
    



    
![png](output_23_353.png)
    



    
![png](output_23_354.png)
    



    
![png](output_23_355.png)
    



    
![png](output_23_356.png)
    



    
![png](output_23_357.png)
    


### Count
has_crossed_line(centroid, gate_y): Fungsi ini mengecek apakah sentroid (titik tengah) objek sudah melewati garis virtual (didefinisikan oleh gate_y). Jika sentroid berada di bawah garis tersebut (lebih besar dari gate_y), maka dianggap objek tersebut sudah melewati garis.

if objectID not in counted_objects: Mengecek apakah objek tersebut sudah dihitung sebelumnya. Jika belum, maka objek tersebut akan dihitung untuk menghindari penghitungan ganda.

Penghitungan kendaraan:
Jika objek yang terdeteksi adalah mobil (car), variabel car_count akan bertambah satu.
Jika objek yang terdeteksi adalah bus (bus), variabel bus_count akan bertambah satu.
Setelah dihitung, objek akan dimasukkan ke dalam counted_objects untuk memastikan tidak dihitung lebih dari sekali.


### Visualisasi

Visualisasi bounding box dan sentroid:
Setiap objek yang terdeteksi (mobil atau bus) akan digambar bounding box di sekelilingnya menggunakan fungsi cv2.rectangle(). Warna bounding box tergantung pada jenis kendaraan:
Merah untuk mobil.
Hijau untuk bus.
Sentroid kendaraan digambar menggunakan cv2.circle(), menampilkan lingkaran di titik tengah objek.
cv2.putText() digunakan untuk menampilkan teks di atas bounding box dan sentroid, menampilkan ID objek dan jenis kendaraan beserta kepercayaan deteksi.

Menampilkan jumlah kendaraan yang telah dihitung:
Jumlah mobil dan bus yang telah melintasi garis ditampilkan di bagian kiri atas frame:
Mobil: Ditampilkan dalam warna merah.
Bus: Ditampilkan dalam warna hijau.

Menampilkan frame:
Setelah semua visualisasi selesai, frame diubah dari BGR ke RGB menggunakan cv2.cvtColor() karena format BGR digunakan oleh OpenCV, sementara Matplotlib menggunakan RGB.
Frame kemudian ditampilkan menggunakan plt.imshow(), tanpa sumbu, karena sumbu dihilangkan dengan plt.axis('off').

## Save Video to MP4


```python
def has_crossed_line(centroid, line_y):
    return centroid[1] > line_y

net, output_layers, classes = load_yolo_model(weights_path, cfg_path, names_path)

video = cv2.VideoCapture(video_path)

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

output_path = "output_toll_gate.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

tracker = CentroidTracker(max_disappeared=5)  
car_count = 0
bus_count = 0
gate_y = points[-1][1] 

counted_objects = set()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    detections = detect_objects(frame, net, output_layers)
    
    rects = [box for (box, class_id, confidence) in detections if classes[class_id] in ["car", "bus"]]
    
    objects = tracker.update(rects)
    
    for objectID, centroid in objects.items():
        vehicle_class = None
        for box, class_id, confidence in detections:
            if classes[class_id] in ["car", "bus"]:
                x, y, w, h = box
                if (x < centroid[0] < x + w) and (y < centroid[1] < y + h):
                    vehicle_class = classes[class_id]
                    break
    
        color = (0, 0, 255) if vehicle_class == "car" else (0, 255, 0) if vehicle_class == "bus" else (255, 255, 255)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if vehicle_class:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{vehicle_class} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if has_crossed_line(centroid, gate_y) and objectID not in counted_objects:
                if vehicle_class == "car":
                    car_count += 1
                elif vehicle_class == "bus":
                    bus_count += 1
                counted_objects.add(objectID)

    frame = draw_gate_lines(frame)
    frame_width = frame.shape[1]
    cv2.putText(frame, "Created by Minan A", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, f"Mobil: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Bus  : {bus_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(frame)
    
video.release()
out.release()

print(f"Total Cars: {car_count}")
print(f"Total Buses: {bus_count}")
```

    Total Cars: 7
    Total Buses: 1
    

Kode ini serupa dengan yang sebelumnya, tetapi dengan tambahan fungsionalitas untuk menyimpan video hasil pengolahan

Kode ini melakukan deteksi, pelacakan, penghitungan kendaraan (mobil dan bus) pada sebuah video, kemudian menampilkan hasilnya secara visual dan menyimpannya ke dalam file video output yang baru dengan nama output_toll_gate.mp4. Hasil penghitungan juga dicetak di akhir untuk memberi tahu total jumlah mobil dan bus yang melintasi garis selama durasi video.
