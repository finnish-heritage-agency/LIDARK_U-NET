# LIDARK_U-NET

## Requirements

Packages:
geopandas                 0.9.0
rasterstats               0.14.0 
tensorflow-gpu            2.6.0
pytorch                   1.9.0 (Optional)

External software:
LAStools (Optional)


## CreateMap

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
