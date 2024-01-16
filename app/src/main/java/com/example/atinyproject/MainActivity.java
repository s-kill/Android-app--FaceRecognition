package com.example.atinyproject;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {


    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    boolean startYolo=false; //Estado de si Yolo está funcionando o no
    boolean isLoaded=false; // Si han sido cargados los pesos


    private Net TinyV3;              //Init de modelo

    List<String> cocoNames = Arrays.asList("person", "bicycle", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "car", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "doughnut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "TV monitor", "laptop", "computer mouse", "remote control", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "pair of scissors", "teddy bear", "hair drier", "toothbrush");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (hasPermission()) {
            cameraBridgeViewBase = (JavaCamera2View)findViewById(R.id.CameraView);
            cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
            cameraBridgeViewBase.setCvCameraViewListener(this);

            baseLoaderCallback = new BaseLoaderCallback(this) {
                @Override
                public void onManagerConnected(int status) {
                    super.onManagerConnected(status);

                    switch(status){

                        case BaseLoaderCallback.SUCCESS:
                            cameraBridgeViewBase.enableView();
                            break;
                        default:
                            super.onManagerConnected(status);
                            break;
                    }
                }
            };

        } else {
            requestPermission();
        }
    }


    public void YOLO(View Button){  //Se presiona boton
        if (!startYolo) {           //Si estaba apagado
            startYolo=true;

        }else{                      //Si esta prendido
            startYolo=false;        //Se apaga
        }
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba(); //Obtencion de 1 frame

        if (startYolo){ //Si YOLO esta activado
            //Pre Procesamiento de Imagen para utilizar el modelo YOLO
            Imgproc.cvtColor(frame,frame,Imgproc.COLOR_RGBA2RGB); //Se cambia de RGBA a RGB
            Mat Crop = Dnn.blobFromImage(frame,0.003922, new Size(416,416), new Scalar(0,0,0), false,false); //resize a 416x416 y se normaliza (1/255=0.003921568....)
            TinyV3.setInput(Crop); //Se indica que este crop es el input

            java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

            List<String> outBlobNames = new java.util.ArrayList<>();
            outBlobNames.add(0, "yolo_16"); //Stride 16
            outBlobNames.add(1, "yolo_23"); //Stride 32

            TinyV3.forward(result, outBlobNames); //Se evalua y se guarda

            float confThreshold = 0.25f; //Umbral de confianza

            List<Integer> clsIds = new ArrayList<>(); //Lista para guardar los Ids
            List<Float> confs = new ArrayList<>();    //Lista para guardar la acc
            List<Rect> rects = new ArrayList<>();     //Lista para guardar los cuadrados a dibujar


            for (int i = 0; i < result.size(); i++){
                Mat output = result.get(i);                              //Output sería el resultado de la capa yolo_16 o 23
                for(int j = 0; j < output.rows(); ++j) {                 // j sería el indice de una caja (Bounding box)
                    Mat box = output.row(j);                             // Se revisa la j-esima caja
                    Mat scores = box.colRange(5, output.cols());         //  Se extrae los scores de las clases [0,1,2,3......79]

                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);   //Se obtiene los minimos y maximos scores (y posiciones)
                    float confidence = (float)mm.maxVal;                //Se obtiene el mejor score
                    Point classIdPoint = mm.maxLoc;                     //Y su posicion

                    if (confidence > confThreshold){                    //Si el scores es mayor que el umbral indicado

                        int centerX = (int)(box.get(0,0)[0] * frame.cols()); //Se obtiene el centro (t_x) del box y se re escala respecto a la imagen original
                        int centerY = (int)(box.get(0,1)[0] * frame.rows()); // t_y
                        int width   = (int)(box.get(0,2)[0] * frame.cols()); //Se obtiene el ancho del box y se re escala respecto a la imagen original
                        int height  = (int)(box.get(0,3)[0] * frame.rows()); //Se obtiene el alto del box y se re escala respecto a la imagen original

                        int left    = centerX - width / 2;  //Se obtiene el punto izquierdo del box
                        int top    = centerY - height / 2;  //Se obtiene el punto superior del box

                        clsIds.add((int)classIdPoint.x);                //Se agrega a la lista el indice del mejor resultado
                        confs.add(confidence);                   //Se agrega a la lista el score del mejor resultado
                        rects.add(new Rect(left, top, width, height));  //Se agrega a la lista el rectangulo a dibujar
                    }
                }
            }

            int Nconfs = confs.size(); //Se calcula la cantidad de los resultados mayores al umbral
            if (Nconfs >= 1){          //Si hay al menos un resultado se aplica NMS (Non-Maximum Suppresion), esto evita varios rectangulos en el mismo objeto
                float nmsThreshold = 0.15f;     //Se establece umbral para NMS
                //Matrices para guardar resultados de NMS
                Rect[] boxesArray = rects.toArray(new Rect[0]);
                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                MatOfInt indices = new MatOfInt();

                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices); //(vector< Rect >, vector< float >,float, const float, vector< int >)

                //Se dibujan los cuadrados finales
                int[] ind = indices.toArray();          //Indices
                for (int i = 0; i < ind.length; ++i){   //Para cada indice
                    int idx = ind[i];                   //
                    Rect box = boxesArray[idx];         //Se obtiene su box

                    int idClass = clsIds.get(idx);      //Se obtiene el nombre de la clase
                    float conf = confs.get(idx);        //Se obtiene la confianza de la prediccion

                    int intConf = (int) (conf * 100);   //Se convierte a %

                    //Se escribe el resultado en top-left de la caja
                    Imgproc.putText(frame, cocoNames.get(idClass)+"/"+intConf+"%",box.tl(),Core.FONT_HERSHEY_SIMPLEX,3,new Scalar(255,0,0),3);
                    //Se dibuja la caja
                    Imgproc.rectangle(frame, box.tl(),box.br(), new Scalar(255,0,0), 3);
                }
            }
        }
        return frame;
    }



    @Override
    public void onCameraViewStarted(int width, int height) {
        if (!isLoaded){
            String TinyCfg = getPath("yolov3-tiny.cfg", this);
            String TinyWeights = getPath("yolov3-tiny.weights", this);
            TinyV3 = Dnn.readNetFromDarknet(TinyCfg, TinyWeights);  //Se carga el modelo
            Toast.makeText(getApplicationContext(),"Modelo Cargado!", Toast.LENGTH_SHORT).show();
            isLoaded=true;
        }


    }

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;

        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            System.out.println(outFile.getAbsolutePath());
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "";
    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    //Comprobar permiso de camara
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)){
                Toast.makeText(getApplicationContext(), "Camera permission are required for this app", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }



}