# LIDARK_U-NET

## About

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

The script uses specified working directory, which should be used to store and organize files and data. The working directory should contain 

## CreateMap

This module provides basic workflow for processing LiDAR data to 2D-visualizations for use with Training and Prediction modules. The module offers one option for processing LiDAR point clouds to elevation models (DTM) and two different options for visualizations (Topographic position index & Multidirectional hillshade). 

Possible arguments include:
* project_folder: path to the project folder ex. 'C:/project/'
* project_area: name of the project area ex. 'Naljanka'. 
* map_type: choose between 'DTM','TPI','MDHS'. 'DTM' requiers lastools path 'C:/lastools/bin/' in the laz2dtm def
* cores: only relevant for lastools. default is 8.
* buffer: the overlap between mapsheets when generating data-maps. Overlaps is needed to get a fully covered area. default is 30. DTM should have a slightly wider buffer then the other layers. When using 30 for TPI, 50 is recommended for DTM.
* pxl_size: the resolution used in generating data-maps. should be the same for all products.
* gpu: If torch is installed, GPU-processing of the TPIs can be done. Else they will be processed by CPU.


CreateMap(project_folder=project_folder,
          project_area='Naljanka',
          map_type='DTM',
          cores=8,
          buffer=50,
          pxl_size=0.25,
          gpu=False)

CreateMap(project_folder=project_folder,
          project_area='Kuivaniemi',
          map_type='TPI',
          cores=1,
          buffer=20,
          pxl_size=0.25,
          gpu=True)

## Training

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

Predict(project_folder=project_folder,
        project_areas=['Naljanka','Kuivaniemi'],
        feature_list=['02_TPI/radius_20/','02_TPI/radius_30/'],
        model_timestamp='20211222-105650',
        test_mapsheets=['##'],
        pred_type='all')


## Evaluate

Evaluate(project_folder=project_folder,
         true_objects='evaluate_true.shp',
         false_objects='evaluate_false.shp',
         pred_type='all',
         eval_aoi=None,
         model_timestamp='20211222-105650',
         eval_name='test1')
