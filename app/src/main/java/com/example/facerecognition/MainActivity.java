package com.example.facerecognition;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.ContextWrapper;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;
import android.content.Context;
import android.content.res.AssetManager;
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
import org.opencv.objdetect.CascadeClassifier; // Classifier

import org.opencv.imgproc.Imgproc;

import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.face.FaceRecognizer;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
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

    boolean startDec=false; //Estado para detectar caras
    boolean startRec=false; //Estado para reconocer caras

    boolean isLoaded=false; // Si han sido cargados los pesos


    private CascadeClassifier face_cascade;  // Init Classifier face
    private CascadeClassifier eye_cascade;  // Init Classifier eye

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    private LBPHFaceRecognizer LBPH_face_rec;
    //private FaceRecognizer face_rec = LBPHFaceRecognizer.create();

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


    public void BD(View Button){  //Se presiona boton
        if (startRec)  {           //Si reconocimiento esta encendido
            Toast.makeText(getApplicationContext(),"Apague Reconocimiento!", Toast.LENGTH_SHORT).show();
        }

        else if (!startDec) {           //Si estaba apagado
            startDec=true;


        }else{                      //Si esta prendido
            startDec=false;        //Se apaga
        }
    }

    public void REC(View Button){  //Se presiona boton
        if (startDec)  {           //Si deteccion esta encendido
            Toast.makeText(getApplicationContext(),"Apague Detección!", Toast.LENGTH_SHORT).show();
        }

        else if (!startRec) {           //Si estaba apagado
            startRec=true;

        }else{                      //Si esta prendido
            startRec=false;        //Se apaga
        }

    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba(); //Obtencion de 1 frame
        Mat frame_gray = new Mat();


        if (startDec){ //Si Deteccion esta activado
            //Pre Procesamiento de Imagen para utilizar el modelo
            Imgproc.cvtColor(frame,frame_gray,Imgproc.COLOR_RGBA2GRAY); //Se cambia de RGBA a GRAY
            Imgproc.equalizeHist(frame_gray,frame_gray);

            MatOfRect faces = new MatOfRect();                     //Matriz de caras
            face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 2, new Size(0, 0), new Size());

            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++)
                Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(255,0,0), 3);
        }

        else if (startRec){ //Si Reconocimiento esta activado
            //Pre Procesamiento de Imagen para utilizar el modelo
            Imgproc.cvtColor(frame,frame_gray,Imgproc.COLOR_RGBA2GRAY); //Se cambia de RGBA a GRAY
            Imgproc.equalizeHist(frame_gray,frame_gray);

            MatOfRect faces = new MatOfRect();                     //Matriz de caras
            face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 2, new Size(0, 0), new Size());

            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++)
                Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(255,0,0), 3);
        }
        return frame;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        if (!isLoaded){

            String frontal_face = getPath("haarcascade_frontalface_default.xml", this);
            String eye_face = getPath("haarcascade_eye.xml", this);



            face_cascade = new CascadeClassifier(frontal_face);
            eye_cascade = new CascadeClassifier(eye_face);

            LBPH_face_rec = LBPHFaceRecognizer.create(0,0);
            //face_rec = LBPHFaceRecognizer.create(5);




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
