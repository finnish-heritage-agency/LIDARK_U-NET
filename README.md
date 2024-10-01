# LIDARK_U-NET

## About

The repository contains code for preparing data and detecting archaeological features from airborne laser scanning data developed in the LIDARK -project (2021-2022). For context see: Anttiroiko et. al. Detecting the Archaeological Traces of Tar Production Kilns in the Northern Boreal Forests Based on Airborne Laser Scanning and Deep Learning, Remote Sensing 2023 15(7) https://doi.org/10.3390/rs15071799

## Dependencies

Python packages and versions used in development environment are listed below. 
* geopandas                 0.9.0
* pandas                    1.3.2 
* numpy                     1.19.5
* rasterstats               0.14.0
* shapely                   1.7.1
* tensorflow                2.6.0
* pytorch                   1.9.0 (Optional)

External software:
* LAStools (Optional)

## Folder structure

The script uses specified working directory, which should be used to store and organize files and data. Most of this structure is created automatically when needed.

The main working directory should contain sub-directories named 'Lidar' and 'DeepLearning'.

The directory 'Lidar' is used to store and organize lidar data. In order to add a lidar dataset, create a new sub-directory, such as 'Lidar/test_dataset'. In order to add lidar point clouds create a sub directory named 00_Original and copy the lidar files there (in .laz or .las format).

The Directory 'DeepLearning' is primarily used to store labels, models, predictions and other information related to the deep-learning process. True and False labels are stored in DeepLearning/train_annotations/

## CreateMap

CreateMap module provides basic workflow for processing LiDAR data to 2D-visualizations for use with Training and Prediction modules. The module offers one option for processing LiDAR point clouds to elevation models (DTM) and two different options for visualizations (Topographic position index & Multidirectional hillshade). 

Possible arguments include:
* project_folder: path to the project folder ex. 'C:/project/'
* project_area: name of the project area ex. 'Naljanka'. 
* map_type: choose between 'DTM','TPI','MDHS'. 'DTM' requiers lastools path 'C:/lastools/bin/' in the laz2dtm def
* cores: only relevant for lastools. default is 8.
* buffer: the overlap between mapsheets when generating data-maps. Overlaps is needed to get a fully covered area. default is 30. DTM should have a slightly wider buffer then the other layers. When using 30 for TPI, 50 is recommended for DTM.
* pxl_size: the resolution used in generating data-maps. should be the same for all products.
* gpu: If torch is installed, GPU-processing of the TPIs can be done. Else they will be processed by CPU.

Example1: Create a DTM of project area 'Naljanka' using LAStools for processing.
CreateMap(project_folder=project_folder,
          project_area='Naljanka',
          map_type='DTM',
          cores=8,
          buffer=50,
          pxl_size=0.25,
          gpu=False)
          
Example2: Create a topographic position index (TPI) from previously created DTM using GPU acceleration (Set gpu to False for CPU version)
CreateMap(project_folder=project_folder,
          project_area='Naljanka',
          map_type='TPI',
          cores=1,
          buffer=20,
          pxl_size=0.25,
          gpu=True)

TPI implementation based on https://github.com/zoran-cuckovic/LandscapeArchaeology.org/blob/master/_posts/2021-12-01-python-tpi.md

## Training

This module is used to builds a training dataset and train models. It can do both in combination or individually, depending on arguments. 

NB! The script expects that filenames of the original .las or .laz files are of fixed length. By default this length is nine characters.

Parameters:
* project_folder: path to the project folder ex. 'C:/project/'
* true_objects: name of shapefile (.shp) with true objects.
* false_objects: name of shapefile (.shp) with false objects.
* feature_list: list of data maps that will be used for creating a training dataset. ex. ['02_TPI/radius_30/'] or ['02_TPI/radius_20/','02_TPI/radius_30/']
* test_mapsheets: define a list with mapsheet names for that should not be included in training. If any names in the list is contained in a mapsheet, it will be excluded. ex. ['S5133G1','S5133E3']
* fetch_data: choose if a training dataset should be created or not.
* data_timestamp: when creating a training dataset, it is saved with a timestamp. define the timestamp to an already existing training dataset if you don't want to generate a new one. (if fetch_data = False)
* augment: if you want to augment the trianing data, to gain a bigger training dataset.
* train_on_data: If you want to use your traning dataset for training.

Training(project_folder=project_folder,
         true_objects='Tar_kilns_update_1.shp',
         false_objects='OtherFeatures_update_1.shp',
         feature_list=['02_TPI/radius_20/','02_TPI/radius_30/'],
         test_mapsheets=['##'],
         fetch_data=False,
         data_timestamp='1221164907',
         augment=True,
         train_on_data=True)



## Predict

This module uses a pre-trained model to predict features in a project area or a selection of test mapsheets. The result is merged into a single shapefile. The output vector result will have both probability and size as attributes. And selecting a model is done by defining a timestamp.

Parameters:
* project_folder: path to the project folder ex. 'C:/project/'
* project_areas: name of the project areas in a list ex. ['Naljanka'] or ['Naljanka','Kuivaniemi']
* feature_list: list of data maps that will be used for prediction. Must be the same as the model is trained on ex. ['02_TPI/radius_30/']
* model_timestamp: the timestamp for an earlier trained model.
* test_mapsheets: a selection of mapsheets for the model to predict on. ex. ['S5133G1','S5133E3']
* pred_type: if the whole area should be predicted, choose 'all'. If only test mapsheets should be predicted, choose 'test'

Example: Predict features from project areas 'Naljanka' and 'Kuivaniemi' using model '20211222-105650'.
Predict(project_folder=project_folder,
        project_areas=['Naljanka','Kuivaniemi'],
        feature_list=['02_TPI/radius_20/','02_TPI/radius_30/'],
        model_timestamp='20211222-105650',
        test_mapsheets=['##'],
        pred_type='all')


## Evaluate
This class evaluates a prediction against an evaluation dataset, stored in '/DeepLearning/evaluate_annotations/', under a given evaluation name folder. The eval_name is used to keep control over different sets of evaluation. (ex. project/DeepLearning/evaluation_annotations/test_evaluation/true_objects.shp) For this evaluation, numbers of true, false, missing and not labeled detections are produced. To help understand the quality of the prediction.

Evaluate(project_folder=project_folder,
         true_objects='evaluate_true.shp',
         false_objects='evaluate_false.shp',
         pred_type='all',
         eval_aoi=None,
         model_timestamp='20211222-105650',
         eval_name='test1')
