# graph_loader_g
C++ Application for real-life testing of CNNs for semantic segmentation and 6-DoF pose estimation.

#Software requirements
1) CUDA v10.0
2) OpenCV
3) StructureCore Camera Driver ( For accessing the camera)
4) a) Tensorflow C Library: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-1.14.0.zip
   b) Extract the tensorflow C library in the "dependencies" folder of the Repo. (Create the dependencies folder first).
   c) Copy the tf_attrtype.h file to "./dependencies/libtensorflow-gpu-windows-x86_64-1.14.0/include/tensorflow/c/"
