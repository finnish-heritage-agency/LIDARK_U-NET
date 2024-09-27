import os
import glob
import rasterio
import gc
import random
import numpy as np
import pandas as pd
import geopandas as gp
import tensorflow as tf
from tqdm import tqdm
from affine import Affine
from shapely.geometry import box
from rasterio.mask import geometry_mask
from datetime import datetime
from rasterstats import zonal_stats
import warnings

warnings.filterwarnings("ignore")

"""
To begin with, 
    place the .las or .laz files in :
            {project_folder} / Lidar / {project_area} / 00_Original /   
    place the false and true label files in :
            {project_folder} / GIS / {project_area} / labels /    
"""


class CreateMap:
    """
    This class creates data-maps used for traning and prediction.

    Parameters
    ----------
    project_folder : path to the project folder ex. 'C:/project/'

    project_area : name of the project area ex. 'Naljanka'

    map_type : choose between 'DTM','TPI','MDHS'.
        'DTM' requiers lastools path 'C:/lastools/bin/' in the laz2dtm def

    cores : only relevant for lastools. default is 8.

    buffer : the overlap between mapsheets when generating data-maps.
        Overlaps is needed to get a fully covered area. default is 30.
        DTM shoud have a slightly wider buffer then the other layers.
        when using 30 for TPI, 50 is recommended for DTM.

    pxl_size : the resolution used in generating data-maps. should be the same for all products.

    gpu : If torch is installed, GPU-processing of the TPIs can be done. Else they will be processed by CPU.

    """

    def __init__(self, project_folder, project_area, map_type, cores=8, buffer=30, pxl_size=0.25, gpu=True):
        self.project_folder = project_folder
        self.project_area = project_area
        self.map_type = map_type
        self.cores = cores
        self.buffer = buffer
        self.pxl_size = pxl_size
        self.gpu = gpu

        # these folders will be generated automatically, except the self.laz_folder.
        self.laz_folder = self.project_folder + '/Lidar/' + self.project_area + '/00_Original'
        self.DTM_folder = self.project_folder + '/Lidar/' + self.project_area + '/01_DTM/'
        self.TPI_folder = self.project_folder + '/Lidar/' + self.project_area + '/02_TPI/'
        self.MDHS_folder = self.project_folder + '/Lidar/' + self.project_area + '/03_MDHS/'

        # execution
        if map_type == 'DTM':
            print('generating DTM from laz')
            print('num cores: ' + str(self.cores))
            self.laz2dtm()
        if map_type == 'TPI':
            print('generating TPI from DTM')
            print('radius: ' + str(self.buffer))
            self.tpi_index()
        if map_type == 'MDHS':
            print('generating multi-directional hillshade')
            self.mdhs()
        else:
            print('map-type not recognized')

    def laz2dtm(self):
        """
        las2DTM generates DTM rasters of all laser data in the self.laz_folder

        :return: DTM rasters in the self.DTM_folder
        """

        # define the path to the lastools bin-folder.
        lastools_path = 'C:/lastools/bin/'

        # creating folder if not already exists
        if not os.path.exists(self.DTM_folder):
            os.makedirs(self.DTM_folder)

        def lasindex(input, cores):
            """
            lasindex creates a lastools call readable for the os.

            lasindex documentation:
                Creates a *.lax file for a given *.las or *.laz file that
                contains spatial indexing information. When this LAX file is
                present it will be used to speed up access to the relevant
                areas of the LAS/LAZ file

            :param input: the folder where las-data is located, defined in __init__ as self.laz_folder
            :param cores: number of cores that should be used for processing
            :return: a os call for lasindex with these parameters
            """
            return lastools_path + \
                   'lasindex ' + \
                   '-i ' + input + '/*.laz ' + \
                   '-cores ' + str(cores)

        def las2dem(input, odir, odix, kill, keep_class, step, nodata, buffer, cores):
            """
            las2dem creates a lastools call readable for the os.

            las2dem documentation:
                This tool reads LIDAR points from the LAS/LAZ format (or some
                ASCII format), triangulates them temporarily into a TIN, and
                then rasters the TIN onto a DEM.

            :param input: the folder where las-data is located, defined in __init__ as self.laz_folder
            :param odir: the folder where DTM-data should be generated, defined in __init__ as self.DTM_folder
            :param odix: output filename suffix
            :param kill: maximum triangle length.
            :param keep_class: laser class filter.
            :param step: resolution of output raster
            :param nodata: nodata value
            :param buffer: buffer size for mapsheets. should be wider than the other products.
            :param cores: number of cores that should be used for processing
            :return: a os call for las2dem with these parameters
            """
            return lastools_path + \
                   'las2dem ' + \
                   '-i ' + input + '/*.laz ' + \
                   '-odir ' + odir + ' ' + \
                   '-otif ' + \
                   '-odix ' + odix + ' ' + \
                   '-elevation ' + \
                   '-kill ' + str(kill) + ' ' + \
                   '-keep_class ' + keep_class + ' ' + \
                   '-step ' + str(step) + ' ' + \
                   '-nodata ' + str(nodata) + ' ' + \
                   '-buffered ' + str(buffer) + ' ' + \
                   '-cores ' + str(cores)

        # runs the call generated from def
        os.system(lasindex(input=self.laz_folder, cores=self.cores))

        # runs the call generated from def
        os.system(las2dem(input=self.laz_folder,
                          odir=self.DTM_folder,
                          odix='_DTM',
                          kill=50,
                          keep_class='2',
                          step=self.pxl_size,
                          nodata=0,
                          buffer=self.buffer,
                          cores=self.cores))

    def tpi_index(self):
        """
        tpi_index generates TPI rasters with a given radius equal the defined self.buffer,
        of all laser data in the self.DTM_folder.
        This algorithm has been optimized to run on GPU.
        If no GPU is availible, CPU will be used.

        TPI documentation:
            https://landscapearchaeology.org/2019/tpi/

        :return: TPI files in the self.TPI_folder
        """

        # creating output-folder if not already exists
        tpi_output = self.TPI_folder + '/radius_' + str(self.buffer) + '/'
        if not os.path.exists(tpi_output):
            os.makedirs(tpi_output)

        def tpi(mx_z, mx_temp, mx_count, range_list):
            """

            :param mx_z: DTM as array
            :param mx_temp: temporary array
            :param mx_count: tracking number of neighbours
            :param range_list: positions to be processed
            :return: TPI output file
            """

            def view(offset_y, offset_x, shape, step=1):
                """
                Function returning two matching numpy views for moving window routines.
                - 'offset_y' and 'offset_x' refer to the shift in relation to the analysed (central) cell
                - 'shape' are 2 dimensions of the data matrix (not of the window!)
                - 'view_in' is the shifted view and 'view_out' is the position of central cells
                (see on LandscapeArchaeology.org/2018/numpy-loops/)
                """
                size_y, size_x = shape
                x, y = abs(offset_x), abs(offset_y)

                x_in = slice(x, size_x, step)
                x_out = slice(0, size_x - x, step)

                y_in = slice(y, size_y, step)
                y_out = slice(0, size_y - y, step)

                # the swapping trick
                if offset_x < 0: x_in, x_out = x_out, x_in
                if offset_y < 0: y_in, y_out = y_out, y_in

                # return window view (in) and main view (out)
                return (y_in, x_in), (y_out, x_out)

            # loop through window and accumulate values
            for (y, x), weight in range_list:

                if weight == 0: continue  # skip zero values !
                # determine views to extract data
                view_in, view_out = view(y - r_y, x - r_x, mx_z.shape)
                # using window weights (eg. for a Gaussian function)
                mx_temp[view_out] += mx_z[view_in] * weight

                # track the number of neighbours
                # (this is used for weighted mean : Σ weights*val / Σ weights)
                mx_count[view_out] += weight

            out = mx_z - mx_temp / mx_count

            return out

        r = self.buffer  # radius for the TPI calculation
        win = np.ones((2 * r + 1, 2 * r + 1))  # filter used as moving window
        r_y, r_x = win.shape[0] // 2, win.shape[1] // 2  # radius of the filter in x and y directions.
        win[r_y, r_x] = 0  # removing the middle pixel
        range_list = []  # fetching positions in the moving window
        for (y, x), weight in np.ndenumerate(win):
            range_list.append([(y, x), weight])

        if self.gpu:
            import torch
            print('GPU processing enabled...')

        proc_files = glob.glob(self.DTM_folder + '/*.tif')  # gathering paths for files to be processed
        for i in tqdm(range(len(proc_files))):
            proc_file = proc_files[i]
            if not os.path.exists(tpi_output + str(os.path.basename(proc_file)[:-7]) + 'TPI.tif'):
                # if not file already exists, DTM raster is opened and read as numpy array.
                dtm_img = rasterio.open(proc_file)
                profile = dtm_img.profile
                dtm_arr = dtm_img.read(1)
                dtm_arr[dtm_arr <= 0] = np.nan  # values lower than 0 in the DTM is beeing ignored.

                # if gpu-processing is enabled, torch converts numpy arrays to tensors that can be stored in GPU memory
                if self.gpu:
                    mx_z = torch.from_numpy(dtm_arr).cuda()
                    mx_temp = torch.zeros(mx_z.shape).cuda()
                    mx_count = torch.zeros(mx_z.shape).cuda()
                else:
                    mx_z = dtm_arr
                    mx_temp = np.zeros(mx_z.shape)
                    mx_count = np.zeros(mx_z.shape)

                out = tpi(mx_z, mx_temp, mx_count, range_list)
                if self.gpu:
                    out = out.cpu()
                    out = np.array(out)

                # converting the float array output to integer
                out = np.nan_to_num(out, nan=0)
                out = out * 100
                out[out > 100] = 100
                out[out < -100] = -100
                out = out.astype(dtype='int16')

                profile['dtype'] = 'int16'
                profile['nodata'] = None

                # cutting away the buffer area as the TPI will generate false values here.
                old_trans = profile['transform']
                buffer_size = int(self.buffer / old_trans[0])
                out = out[buffer_size:-buffer_size, buffer_size:-buffer_size]

                # updating metadata for the new image size and positioning
                profile['height'] = out.shape[0]
                profile['width'] = out.shape[1]
                profile['transform'] = Affine(old_trans[0], old_trans[1], old_trans[2] + self.buffer, old_trans[3],
                                              old_trans[4], old_trans[5] - self.buffer)

                # the CPU memory can sometimes be overloaded with waste when switching between CPU and GPU
                # therefore this will clean up memory waste.
                if self.gpu:
                    gc.collect()
                    torch.cuda.empty_cache()

                # tries to export as geotiff
                try:
                    with rasterio.open(tpi_output + str(os.path.basename(proc_file)[:-7]) + 'TPI.tif', 'w',
                                       **profile) as dst:
                        dst.write(out, 1)
                except:
                    print("cannot export file")

    def mdhs(self):
        """
        MDHS (Multi-directional hillshade) is a hillshade product that combines several hillshade directions
        for optimal insight in the terrain. The lowest value at each pixle for each direction is kept.

        :return: MDHS files in the self.MDHS_folder
        """

        # creating folder if not already exists
        if not os.path.exists(self.MDHS_folder):
            os.makedirs(self.MDHS_folder)

        # predefined hillshade directions and altitude. Can be edited.
        directions = [330, 270, 210, 150, 90, 30]
        angle_altitude = 45

        imgs = glob.glob(self.DTM_folder + '/*.tif')  # gathering paths for files to be processed
        for i in range(len(imgs)):
            # DTM raster is opened and read as numpy array.
            dtm_img = rasterio.open(imgs[i])
            dtm_arr = dtm_img.read(1)
            dtm_arr[dtm_arr <= 0] = np.nan

            # creating empty hillshade array with one band per direction
            mdhs_arr = np.zeros(shape=(len(directions), dtm_arr.shape[0], dtm_arr.shape[1]))
            for j in tqdm(range(len(directions))):
                # for each direction, a hillshade is created
                azimuth = directions[j]
                x, y = np.gradient(dtm_arr)
                slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
                aspect = np.arctan2(-x, y)
                azimuthrad = azimuth * np.pi / 180.
                altituderad = angle_altitude * np.pi / 180.
                shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(
                    (azimuthrad - np.pi / 2.) - aspect)

                # each hillshade is added to a band
                mdhs_arr[j, :, :] = shaded

            # converting the float array output to integer
            mdhs_arr = mdhs_arr.min(axis=0)
            mdhs_arr = np.nan_to_num(mdhs_arr, nan=0)
            mdhs_arr = 255 * mdhs_arr
            mdhs_arr = mdhs_arr.astype(dtype='uint8')

            # updating metadata for the new image
            profile = dtm_img.profile
            profile['count'] = 1
            profile['dtype'] = 'uint8'
            profile['nodata'] = 0

            # tries to export as geotiff
            try:
                with rasterio.open(self.MDHS_folder + str(os.path.basename(imgs[i])[:-11]) + '_MDHS.tif', 'w',
                                   **profile) as dst:
                    dst.write(mdhs_arr, 1)
            except:
                print("cannot export file")


class Training:
    """
    This class both builds a training dataset, and trains a model. It can do both in combination, or seperated if
    a training dataset exists. After the update, this is Class is no longer connnected to a project area. True and
    false objects are instead stored under /DeepLearning/train_annotations/'.

    If the gpu-version of tensorflow is installed, and it is able to use the GPU in the machine, training will be
    done using GPU memory.

    Parameters
    ----------
    project_folder : path to the project folder ex. 'C:/project/'

    true_objects : name of shapefile (.shp) with true objects.

    false_objects : name of shapefile (.shp) with false objects.

    feature_list : list of data maps that will be used for creating a training dataset. ex. ['02_TPI/radius_30/'] or
        ['02_TPI/radius_20/','02_TPI/radius_30/']

    test_mapsheets : define a list with mapsheet names for that should not be included in training.
        If any names in the list is contained in a mapsheet, it will be excluded. ex. ['S5133G1','S5133E3']

    fetch_data : choose if a training dataset should be created or not.

    data_timestamp : when creating a training dataset, it is saved with a timestamp. define the timestamp to a already
        existing training dataset if you don't want to generate a new one. (if fetch_data = False)

    augment : if you want to augment the trianing data, to gain a bigger training dataset.

    train_on_data : If you want to use your traning dataset for training.

    """

    def __init__(self, project_folder, true_objects, false_objects, feature_list,
                 test_mapsheets, fetch_data=True, data_timestamp=None, augment=True, train_on_data=True):
        self.project_folder = project_folder
        self.true_objects = true_objects
        self.false_objects = false_objects
        self.feature_list = feature_list
        self.test_mapsheets = test_mapsheets
        self.fetch_data = fetch_data
        self.data_timestamp = data_timestamp
        self.augment = augment
        self.train_on_data = train_on_data

        # these folders will be generated automatically
        self.training_folder = self.project_folder + '/DeepLearning/'
        self.label_folder = self.training_folder + '/train_annotations/'
        self.log_folder = self.training_folder + '/logs/'

        # the EPSG code can be changed, but should be the same as the laser-data.
        self.EPSG_code = 'EPSG:3067'

        # length of mapsheet names
        self.mapsheet_digits = 9

        # training hyper-parameters. Can be edited.
        self.img_size = (512, 512)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.val_split = 0.2
        self.max_epochs = 200
        self.unet_size = 16
        self.dropout = 0.0

        # if not a training dataset is created, use fetch data to create one. Else, use the timestamp to locate it.
        if self.fetch_data:
            self.samples, self.labels = self.create_samples()
            if self.augment:
                self.augment_data()
            self.save_training_data()
        else:
            self.samples, self.labels = self.load_training_data()

        # data being shuffled for a optimal training situation
        self.train_samples, self.train_labels, self.val_samples, self.val_labels = self.shuffle_data()

        # start training
        if self.train_on_data:
            self.train_model()
            self.create_log()

    def create_samples(self):
        """
        Creates the training dataset based on the data maps, and labels defined

        :return:  a numpy array with samples, and one with labels.
        """
        # reading the labels as vector data
        true_objects = gp.read_file(self.label_folder + '/' + self.true_objects)
        false_objects = gp.read_file(self.label_folder + '/' + self.false_objects)

        # gathering information about the label positions in the data maps
        imgs = glob.glob(self.project_folder + '/Lidar/*/' + self.feature_list[0] + '/*.tif')
        training_meta = pd.DataFrame(columns=['mapsheet', 'y_min', 'y_max', 'x_min', 'x_max'])
        for i in tqdm(range(len(imgs))):
            # selecting a image to gather information about its positioning
            img = rasterio.open(imgs[i])
            mapsheet_name = os.path.basename(imgs[i])[:self.mapsheet_digits]

            # checking to see if the selected image is in the test-list
            train = True
            for test in self.test_mapsheets:
                if test in mapsheet_name:
                    train = False
            if train:
                # define the boundaries of the image
                img_bounds = gp.GeoDataFrame(
                    geometry=[box(img.bounds[0], img.bounds[1], img.bounds[2], img.bounds[3])], crs=self.EPSG_code)

                # clipping the vector labels to the image bounds
                true_temp = gp.clip(true_objects, img_bounds)
                false_temp = gp.clip(false_objects, img_bounds)

                # if the image contains any true labels, 2 random snapshot positions are stored for each label
                if len(true_temp) > 0:
                    for j in range(len(true_temp)):
                        object_bounds = true_temp.iloc[j].geometry.bounds
                        min_minx = object_bounds[2] - (self.img_size[1] * img.transform[0])
                        max_minx = object_bounds[0]
                        min_maxy = object_bounds[3]
                        max_maxy = object_bounds[1] + (self.img_size[0] * img.transform[0])
                        for k in range(2):
                            minx = random.randint(int(min_minx), int(max_minx))
                            maxy = random.randint(int(min_maxy), int(max_maxy))
                            x_arr_min = int((minx - img.transform[2]) / img.transform[0])
                            y_arr_min = int((img.transform[5] - maxy) / img.transform[0])
                            x_arr_max = int(x_arr_min + self.img_size[1])
                            y_arr_max = int(y_arr_min + self.img_size[0])
                            data = [mapsheet_name, y_arr_min, y_arr_max, x_arr_min, x_arr_max]
                            training_meta = training_meta.append(
                                pd.DataFrame([data], columns=['mapsheet', 'y_min', 'y_max', 'x_min', 'x_max']))

                # if the image contains any false labels, 2 random snapshot positions are stored for each label
                if len(false_temp) > 0:
                    for j in range(len(false_temp)):
                        object_bounds = false_temp.iloc[j].geometry.bounds
                        min_minx = object_bounds[2] - (self.img_size[1] * img.transform[0])
                        max_minx = object_bounds[0]
                        min_maxy = object_bounds[3]
                        max_maxy = object_bounds[1] + (self.img_size[0] * img.transform[0])
                        for k in range(2):
                            minx = random.randint(int(min_minx), int(max_minx))
                            maxy = random.randint(int(min_maxy), int(max_maxy))
                            x_arr_min = int((minx - img.transform[2]) / img.transform[0])
                            y_arr_min = int((img.transform[5] - maxy) / img.transform[0])
                            x_arr_max = int(x_arr_min + self.img_size[1])
                            y_arr_max = int(y_arr_min + self.img_size[0])
                            data = [mapsheet_name, y_arr_min, y_arr_max, x_arr_min, x_arr_max]
                            training_meta = training_meta.append(
                                pd.DataFrame([data], columns=['mapsheet', 'y_min', 'y_max', 'x_min', 'x_max']),
                                ignore_index=True)
            else:
                print('skipping test mapsheet')

        # two arrays are being created to contain data from the data maps, and the co-responding label.
        labels = np.empty(shape=(len(training_meta), self.img_size[0], self.img_size[1]), dtype='uint8')
        samples = np.empty(shape=(len(training_meta), self.img_size[0], self.img_size[1], len(self.feature_list)),
                           dtype='int8')

        # finding all mapsheets that will be included in generating training data
        train_mapsheets = training_meta['mapsheet'].unique()

        # iterating over the label positions, creating arrays for each
        label_index = 0
        for i in tqdm(range(len(train_mapsheets))):
            mapsheet = train_mapsheets[i]
            local_samples = training_meta.loc[training_meta['mapsheet'] == mapsheet]
            mask_img = \
                glob.glob(self.project_folder + '/Lidar/*/' + self.feature_list[
                    0] + '/' + mapsheet + '*.tif')[0]
            mask_img = rasterio.open(mask_img)
            x_clip_max = mask_img.shape[1]
            y_clip_max = mask_img.shape[0]
            label_mask = geometry_mask(true_objects.geometry, mask_img.shape, mask_img.transform, all_touched=True,
                                       invert=True)
            for j in range(len(local_samples)):
                xmin = np.clip(local_samples['x_min'].iloc[j], 0, x_clip_max)
                ymin = np.clip(local_samples['y_min'].iloc[j], 0, y_clip_max)
                xmax = np.clip(local_samples['x_max'].iloc[j], 0, x_clip_max)
                ymax = np.clip(local_samples['y_max'].iloc[j], 0, y_clip_max)

                label = label_mask[ymin:ymin + self.img_size[0], xmin:xmin + self.img_size[1]]
                if not label.shape == self.img_size:
                    label = label_mask[ymax - self.img_size[0]:ymax, xmax - self.img_size[1]:xmax]
                if not label.shape == self.img_size:
                    label = label_mask[ymin:ymin + self.img_size[0], xmax - self.img_size[1]:xmax]
                if not label.shape == self.img_size:
                    label = label_mask[ymax - self.img_size[0]:ymax, xmin:xmin + self.img_size[1]]

                labels[label_index, :, :] = label
                label_index += 1

        # iterating over all data maps. creating arrays for each position and stacking data maps if more than one.
        for k in range(len(self.feature_list)):
            feature_index = 0
            for i in tqdm(range(len(train_mapsheets))):
                mapsheet = train_mapsheets[i]
                local_samples = training_meta.loc[training_meta['mapsheet'] == mapsheet]
                feature_img = glob.glob(
                    self.project_folder + '/Lidar/*/' + self.feature_list[k] + '/' +
                    mapsheet + '*.tif')[0]
                feature_img = rasterio.open(feature_img)
                x_clip_max = feature_img.shape[1]
                y_clip_max = feature_img.shape[0]
                feature_arr = feature_img.read(1)
                feature_arr = feature_arr.astype(dtype='int8')
                for j in range(len(local_samples)):
                    xmin = np.clip(local_samples['x_min'].iloc[j], 0, x_clip_max)
                    ymin = np.clip(local_samples['y_min'].iloc[j], 0, y_clip_max)
                    xmax = np.clip(local_samples['x_max'].iloc[j], 0, x_clip_max)
                    ymax = np.clip(local_samples['y_max'].iloc[j], 0, y_clip_max)

                    feature = feature_arr[ymin:ymin + self.img_size[0], xmin:xmin + self.img_size[1]]
                    if not feature.shape == self.img_size:
                        feature = feature_arr[ymax - self.img_size[0]:ymax, xmax - self.img_size[1]:xmax]
                    if not feature.shape == self.img_size:
                        feature = feature_arr[ymin:ymin + self.img_size[0], xmax - self.img_size[1]:xmax]
                    if not feature.shape == self.img_size:
                        feature = feature_arr[ymax - self.img_size[0]:ymax, xmin:xmin + self.img_size[1]]

                    samples[feature_index, :, :, k] = feature
                    feature_index += 1

        return samples, labels

    def augment_data(self):
        """
        Augmenting the dataset generated in create_samples. Flipping the images aand labels horizontally and vertically

        :return: an extended version of the training dataset
        """
        # iterating thorugh the training dataset
        aug_samples = np.empty(
            shape=(len(self.samples), self.img_size[0], self.img_size[1], len(self.feature_list)), dtype='int8')
        aug_labels = np.empty(
            shape=(len(self.labels), self.img_size[0], self.img_size[1]), dtype='uint8')
        for i in tqdm(range(len(self.samples))):
            # horizontal flip
            X_lr = np.fliplr(np.copy(self.samples[i]))
            y_lr = np.fliplr(np.copy(self.labels[i]))
            aug_samples[i, :, :, :] = X_lr
            aug_labels[i, :, :] = y_lr
        self.samples = np.append(self.samples, aug_samples, axis=0)
        self.labels = np.append(self.labels, aug_labels, axis=0)

        aug_samples = np.empty(
            shape=(len(self.samples), self.img_size[0], self.img_size[1], len(self.feature_list)), dtype='int8')
        aug_labels = np.empty(
            shape=(len(self.labels), self.img_size[0], self.img_size[1]), dtype='uint8')
        for i in tqdm(range(len(self.samples))):
            # vertical flip
            X_ud = np.flipud(np.copy(self.samples[i]))
            y_ud = np.flipud(np.copy(self.labels[i]))
            aug_samples[i, :, :, :] = X_ud
            aug_labels[i, :, :] = y_ud

        self.samples = np.append(self.samples, aug_samples, axis=0)
        self.labels = np.append(self.labels, aug_labels, axis=0)

    def save_training_data(self):
        """
        Saves the training dataset to.npy files with a timestamp generated at the actual time of saving.

        :return: two .npy files. One for samples and one for labels.
        """
        self.data_timestamp = datetime.now().strftime('%m%d%H%M%S')
        samples_folder = self.training_folder + '/samples/'
        labels_folder = self.training_folder + '/labels/'
        if not os.path.exists(samples_folder):
            os.makedirs(samples_folder)
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)
        np.save(samples_folder + 'samples_' + self.data_timestamp + '.npy', self.samples)
        np.save(labels_folder + 'labels_' + self.data_timestamp + '.npy', self.labels)

    def load_training_data(self):
        """
        Loads the samples and labels from .npy files at the selected timestamp.

        :return: samples and lables as numpy arrays
        """
        samples = np.load(self.training_folder + '/samples/samples_' + self.data_timestamp + '.npy')
        labels = np.load(self.training_folder + '/labels/labels_' + self.data_timestamp + '.npy')

        return samples, labels

    def shuffle_data(self):
        """
        Shuffels the training dataset for an optimal training situation.
        Also splits the data into training and validation datasets. Based on the selected ratio at self.val_split

        :return: samples and labels for training and validation data.
        """
        idx = list(range(len(self.samples)))
        random.shuffle(idx)
        split = int(len(self.samples) * self.val_split)
        train_idx = idx[:-split]
        val_idx = idx[-split:]

        train_samples = self.samples[train_idx]
        train_labels = self.labels[train_idx]
        val_samples = self.samples[val_idx]
        val_labels = self.labels[val_idx]

        return train_samples, train_labels, val_samples, val_labels

    def train_model(self):
        """
        Trains a model based on the training and validation data.
        Saves the best model to a .h5 file at a new timestamp folder.
        This folder will also be where predictions with the given model is stored.
        Several hyper-parameters can be manually tuned in the __init__ def

        :return: a trained model ready for predictions
        """

        # creating a directory for the model output
        self.model_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = self.training_folder + 'model/' + self.model_timestamp
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # defining the model
        model = self.get_unet_model((self.img_size[0], self.img_size[1], len(self.feature_list)),
                                    1, 'relu', self.dropout)

        # choosing an optimizer for the model
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # compiling the model
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[self.iou])

        # A checkpointer that saves only the best model based on validation results for each epoch
        model_checkpointer_callback = tf.keras.callbacks.ModelCheckpoint(model_dir + '/model.h5',
                                                                         monitor='val_iou',
                                                                         save_best_only=True,
                                                                         verbose=1, mode='max')

        # Stops the training if the model reaches a performance roof
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_iou', patience=10, mode='max')

        # training the model
        history = model.fit(self.train_samples,
                            self.train_labels,
                            batch_size=self.batch_size,
                            epochs=self.max_epochs,
                            callbacks=[model_checkpointer_callback, early_stopping_callback],
                            validation_data=(self.val_samples, self.val_labels))

        self.best_val = max(history.history['val_iou'])

    def pool_encoder_block(self, x, conv_block, features, block_name, activation):
        """
        Encoder blocks using max pooling
        """

        x = conv_block(x, features, block_name, activation)
        p = tf.keras.layers.MaxPool2D((2, 2), name=f"{block_name}_down_pool")(x)

        return x, p

    def simple_conv_block(self, x, features, block_name, activation):
        """
        Create conv2D - BN - act - conv2D - BN - act block.
        A simple convolution block with two convolution layers and batch normalization.
        """

        # Two 3x3 convolutions
        x = self.conv_bn_act(x, features, f"{block_name}_c1", 3, 1, activation)
        x = self.conv_bn_act(x, features, f"{block_name}_c2", 3, 1, activation)

        return x

    def conv_bn_act(self, x, features, name, kernel=1, stride=1, activation=None):
        """
        Basic building block of Convolution - Batch normalization - Activation
        """

        # 3x3 convolution layer without bias, as we have learned gamma - beta parameters in the BN
        x = tf.keras.layers.Conv2D(features, kernel, stride,
                                   padding="same",
                                   name=name + "_conv",
                                   use_bias=False,
                                   data_format="channels_last")(x)

        # Batch normalization, with learned bias
        x = tf.keras.layers.BatchNormalization(name=name + "_batchnorm")(x)

        # Activation
        if activation:
            x = tf.keras.layers.Activation(activation, name=name + "_activation")(x)

        return x

    def decoder_block_addskip(self, input, skip_features, conv_block, features, block_name, activation, dropout=0.0):
        # Upscale convolution
        x = tf.keras.layers.Conv2DTranspose(
            features, (3, 3), strides=2, name=block_name + "_conv_up", padding="same")(input)

        # Add in skip connection
        x = tf.keras.layers.Add(name=block_name + "_add_skip")([x, skip_features])

        # Dropout
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=block_name + "_drop")(x)

        # Convolution block
        x = conv_block(x, features, block_name, activation)

        return x

    def get_unet_model(self, img_size, num_classes, activation_function, dropout=0.0):
        inputs = tf.keras.layers.Input(img_size) #TODO fix input potblem
        features = self.unet_size

        # Downsample
        s1, p1 = self.pool_encoder_block(inputs, self.simple_conv_block, features, 'encoder1',
                                         activation_function)
        s2, p2 = self.pool_encoder_block(p1, self.simple_conv_block, features * 2, 'encoder2',
                                         activation_function)
        s3, p3 = self.pool_encoder_block(p2, self.simple_conv_block, features * 4, 'encoder3',
                                         activation_function)
        s4, p4 = self.pool_encoder_block(p3, self.simple_conv_block, features * 8, 'encoder4',
                                         activation_function)

        # Final downsampled block
        b1 = self.simple_conv_block(p4, features * 16, 'b1', activation_function)

        # Upsample
        d1 = self.decoder_block_addskip(b1, s4, self.simple_conv_block, features * 8, 'decoder1',
                                        activation_function, dropout)
        d2 = self.decoder_block_addskip(d1, s3, self.simple_conv_block, features * 4, 'decoder2',
                                        activation_function, dropout)
        d3 = self.decoder_block_addskip(d2, s2, self.simple_conv_block, features * 2, 'decoder3',
                                        activation_function, dropout)
        d4 = self.decoder_block_addskip(d3, s1, self.simple_conv_block, features, 'decoder4',
                                        activation_function, dropout)

        # Add a per-pixel classification layer
        outputs = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid")(d4)
        model = tf.keras.Model(inputs, outputs, name="U-Net")

        return model

    def iou(self, y_true, y_pred):
        """
        Intersect over union (IoU) used for validation metric in the training process
        """

        labels = y_true
        predictions = y_pred

        # Perform threshold (used to convert fuzzy results to mask, at a given threshold)
        predictions_thresholded = tf.cast(predictions > 0.5, tf.int32)

        # defining the prediction and the labels for overlap
        labels_c = tf.keras.backend.cast(tf.keras.backend.equal(labels, 1), tf.keras.backend.floatx())
        pred_c = tf.keras.backend.cast(tf.keras.backend.equal(predictions_thresholded, 1), tf.keras.backend.floatx())

        # setting the maxim values
        labels_c_sum = tf.keras.backend.sum(labels_c)
        pred_c_sum = tf.keras.backend.sum(pred_c)

        # calculating IoU
        intersect = tf.keras.backend.sum(labels_c * pred_c)
        union = labels_c_sum + pred_c_sum - intersect
        iou = intersect / union

        return iou

    def create_log(self):
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        log_df = pd.DataFrame(index=['log_id', 'data_id', 'epsg', 'num samples', 'augmented',
                                     'features', 'img size', 'learning rate', 'batch size',
                                     'validation split', 'max epochs', 'unet size', 'dropout', 'best iou'])
        #TODO fill
        log_df.loc['log_id', 'values'] = self.model_timestamp
        log_df.loc['data_id','values'] = self.data_timestamp
        log_df.loc['epsg', 'values'] = self.EPSG_code
        log_df.loc['num samples', 'values'] = len(self.samples)
        log_df.loc['augmented', 'values'] = self.augment
        log_df.loc['features', 'values'] = self.feature_list
        log_df.loc['img size', 'values'] = self.img_size
        log_df.loc['learning rate', 'values'] = self.learning_rate
        log_df.loc['batch size', 'values'] = self.batch_size
        log_df.loc['validation split', 'values'] = self.val_split
        log_df.loc['max epochs', 'values'] = self.max_epochs
        log_df.loc['unet size', 'values'] = self.unet_size
        log_df.loc['dropout', 'values'] = self.dropout
        log_df.loc['best iou', 'values'] = self.best_val

        log_df.to_csv(self.log_folder + '/' + self.model_timestamp + '.csv')


class Predict:
    """
    This class predicts an project area, or a selection of test mapsheets.
    It also combines the result into a merged shapefile.
    The output vector result will have both probability and size as attributes.

    An already trained model is needed. And selecting a model is done by defining a timestamp.

    Parameters
    ----------
    project_folder : path to the project folder ex. 'C:/project/'

    project_areas : name of the project areas in a list ex. ['Naljanka'] or ['Naljanka','Kuivaniemi']

    feature_list : list of data maps that will be used for prediction. Must be the same as the model is trained on
        ex. ['02_TPI/radius_30/']

    model_timestamp : the timestamp for an erlier trained model.

    test_mapsheets : a selection of mapsheets for the model to predict on. ex. ['S5133G1','S5133E3']

    pred_type : if the whole area should be predicted, choose 'all'. If only test mapsheets should be predicted,
        choose 'test'

    """

    def __init__(self, project_folder, project_areas, feature_list, model_timestamp, test_mapsheets, pred_type='all'):
        self.project_folder = project_folder
        self.project_areas = project_areas
        self.feature_list = feature_list
        self.model_timestamp = model_timestamp
        self.test_mapsheets = test_mapsheets
        self.pred_type = pred_type

        # these folders will be generated automatically
        self.model_folder = self.project_folder + '/DeepLearning/model/'

        # the EPSG code can be changed, but should be the same as the laser-data.
        self.EPSG_code = 'EPSG:3067'

        # training hyper-parameters. Can be edited. But needs to be the same as in the training class.
        self.img_size = (512, 512)

        # loading model
        self.model = self.load_model()

        # predicting on mapsheets
        if pred_type == 'all':
            # looping through the selected project areas
            for i in range(len(project_areas)):
                # a merge-list is defined to append every predicted mapsheet path. This to merge all in the end
                merge_list = []
                # locating all mapsheets in the selected area ready for prediction
                pred_list = glob.glob(self.project_folder + '/Lidar/' + project_areas[i] + '/' +
                                      self.feature_list[0] + '/*.tif')
                if len(pred_list) == 0:
                    print('No mapsheets found. Check if feature maps have been generated for ' + self.feature_list[0])
                # predicting mapsheets
                for j in range(len(pred_list)):
                    ms_name = os.path.basename(pred_list[j])[:9]
                    print(ms_name)
                    try:
                        merge_list.append(self.pred_mapsheet(ms_name, project_areas[i]))
                    except:
                        print('could not predict mapsheet')

                # merging results to a merged vector output
                self.merge_pred_shapes(project_areas[i], pred_type, merge_list)

        elif pred_type == 'test':
            # looping through the selected project areas
            for i in range(len(project_areas)):
                # a merge-list is defined to append every predicted mapsheet path. This to merge all in the end
                merge_list = []
                # looping through the selected test mapsheets
                for j in range(len(self.test_mapsheets)):
                    test_mapsheet = self.test_mapsheets[j]
                    pred_list = glob.glob(self.project_folder + '/Lidar/' + project_areas[i] + '/' +
                                          self.feature_list[0] + '/' + test_mapsheet + '*.tif')
                    if len(pred_list) == 0:
                        print(
                            'No mapsheets found. Check if feature maps have been generated for ' + self.feature_list[0])
                    for k in range(len(pred_list)):
                        ms_name = os.path.basename(pred_list[k])[:9]
                        print(ms_name)
                        try:
                            merge_list.append(self.pred_mapsheet(ms_name, project_areas[i]))
                        except:
                            print('could not predict mapsheet')

                # merging results to a merged vector output
                self.merge_pred_shapes(project_areas[i], pred_type, merge_list)
        else:
            print('type "all" or "test" in "pred_type" to predict')

    def pred_mapsheet(self, ms_name, project_area):
        """
        predicts a given mapsheet with a trained model.

        :param ms_name: the name of the mapsheet being predicted
        :return: both a raster and a vector prediction of the given mapsheet.
        """

        # gathering metadata for the actual mapsheet
        height_list = []
        width_list = []
        for i in range(len(self.feature_list)):
            pred_meta = rasterio.open(glob.glob(self.project_folder + '/Lidar/' + project_area + '/' +
                                            self.feature_list[i] + '/' + ms_name + '*.tif')[0])
            height_list.append(pred_meta.height)
            width_list.append(pred_meta.width)

        pred_height = min(height_list)
        pred_width = min(width_list)

        # defining the array where data maps are stored for prediction
        pred_arr = np.empty(shape=(pred_height, pred_width, len(self.feature_list)), dtype='int8')

        # defining output paths
        pred_output_path = self.model_folder + self.model_timestamp + '/predictions/'
        raster_output_path = pred_output_path + project_area + '/raster/'
        shape_output_path = pred_output_path + project_area + '/shapes/'

        if not os.path.exists(raster_output_path + ms_name + '.tif'):

            # fetching data from each data map into the array for prediction
            for i in range(len(self.feature_list)):
                feature_name = self.feature_list[i]
                feature_path = glob.glob(self.project_folder + '/Lidar/' + project_area + '/' +
                                         feature_name + '/' + ms_name + '*.tif')[0]
                feature_img = rasterio.open(feature_path)
                feature_arr = feature_img.read(1)
                feature_arr = feature_arr.astype(dtype='int8')
                height_cut = int((feature_arr.shape[0] - pred_height) / 2)
                width_cut = int((feature_arr.shape[1] - pred_width) / 2)
                if height_cut > 0:
                    feature_arr = feature_arr[height_cut:-height_cut,:]
                if width_cut > 0:
                    feature_arr = feature_arr[:,width_cut:-width_cut]
                pred_arr[:, :, i] = feature_arr

            # defining looping parameters based on the size of the input data maps.
            y_start = int(self.img_size[0] / 2)
            y_stop = int(pred_height - self.img_size[0] / 2)
            y_step = int(self.img_size[0] / 4)
            x_start = int(self.img_size[1] / 2)
            x_stop = int(pred_width - self.img_size[1] / 2)
            x_step = int(self.img_size[1] / 4)

            # cutting and stacking images from the data maps
            pred_samples = np.zeros(shape=(0, self.img_size[0], self.img_size[1], len(self.feature_list)), dtype='int8')
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    pred_sample = pred_arr[int(y - (self.img_size[0] / 2)):int(y + (self.img_size[0] / 2)),
                                  int(x - (self.img_size[1] / 2)):int(x + (self.img_size[1] / 2)), :]
                    pred_samples = np.append(pred_samples, np.expand_dims(pred_sample, 0), axis=0)

            # prediction the images gathered from the data maps
            pred_samples = self.model.predict(pred_samples)

            # putting predictions back to it's original position
            prob_arr = np.zeros(shape=(pred_height, pred_width))
            i = 0
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    y_0 = int(y - (self.img_size[0] / 2))
                    y_1 = int(y + (self.img_size[0] / 2))
                    x_0 = int(x - (self.img_size[1] / 2))
                    x_1 = int(x + (self.img_size[1] / 2))
                    init_arr = prob_arr[y_0:y_1, x_0:x_1]
                    pred_arr = pred_samples[i, :, :, 0]

                    # the highest probability value is used per pixel
                    mask_arr = (pred_arr > init_arr).astype(int)
                    pred_arr = pred_arr * mask_arr
                    init_arr = init_arr * (mask_arr == 0).astype(int)
                    pred_arr = pred_arr + init_arr

                    prob_arr[y_0:y_1, x_0:x_1] = pred_arr
                    i += 1

            # masking out low probability pixels
            pred_arr = prob_arr > 0.5

            # converting mask to integer
            pred_arr = (pred_arr > 0).astype(dtype='uint8')

            if pred_arr.max() > 0:
                # converting raster to vector
                shapes = rasterio.features.shapes(pred_arr, connectivity=4, transform=pred_meta.transform)
                records = [{"geometry": geometry, "properties": {"value": value}}
                           for (geometry, value) in shapes if value == 1]
                geoms = list(records)
                polygons = gp.GeoDataFrame.from_features(geoms, crs=self.EPSG_code)

                # simplifying the polygons
                polygons['geometry'] = polygons.geometry.simplify(tolerance=0.25)

                if not os.path.exists(shape_output_path):
                    os.makedirs(shape_output_path)

                # applying a mean probability value for each polygon
                prob_values = zonal_stats(polygons, prob_arr, affine=pred_meta.transform, stats=['mean'])
                polygons['prob'] = pd.DataFrame(prob_values)['mean']
                polygons['area'] = polygons.geometry.area

                # exporting the shapefile
                polygons.to_file(shape_output_path + ms_name + '.shp')

            # updating raster metadata
            profile = pred_meta.profile
            profile['dtype'] = 'uint8'
            profile['nodata'] = None

            if not os.path.exists(raster_output_path):
                os.makedirs(raster_output_path)

            # exporting raster output as geotiff
            with rasterio.open(raster_output_path + ms_name + '.tif', 'w', **profile) as dst:
                dst.write(pred_arr, 1)

        return shape_output_path + ms_name + '.shp'

    def load_model(self):
        """loads the model to be used for prediction"""
        model_path = self.model_folder + self.model_timestamp + '/model.h5'

        dependencies = {
            'iou': self.iou
        }

        model = tf.keras.models.load_model(model_path, custom_objects=dependencies)

        return model

    def merge_pred_shapes(self, project_area, sufflix, input_list):
        """merging shapefiles and fixing overlap zones"""

        # appending all shapes to one table
        merged_shapes = gp.GeoDataFrame()
        for i in tqdm(range(len(input_list))):
            if os.path.exists(input_list[i]):
                merged_shapes = merged_shapes.append(gp.read_file(input_list[i]))

        # merging shapes that overlap between the mapsheets
        merged_geos = gp.GeoDataFrame(merged_shapes.geometry.unary_union)
        merged_geos['geo_id'] = merged_geos.index
        merged_geos = gp.GeoDataFrame(merged_geos, geometry=merged_geos[0])
        merged_geos = merged_geos[['geo_id', 'geometry']]
        merged_shapes = gp.overlay(merged_shapes, merged_geos)
        merged_shapes = merged_shapes.dissolve(by='geo_id')

        # exporting the merged shapefile
        merged_shapes.crs = self.EPSG_code
        merged_shapes.to_file(self.model_folder + self.model_timestamp + '/predictions/' + project_area + '/merged_shapes_' + sufflix + '.shp')

    def iou(self, y_true, y_pred):
        """
        Intersect over union (IoU) used for validation metric in the training process
        """

        labels = y_true
        predictions = y_pred

        # Perform threshold (used to convert fuzzy results to mask, at a given threshold)
        predictions_thresholded = tf.cast(predictions > 0.5, tf.int32)

        # defining the prediction and the labels for overlap
        labels_c = tf.keras.backend.cast(tf.keras.backend.equal(labels, 1), tf.keras.backend.floatx())
        pred_c = tf.keras.backend.cast(tf.keras.backend.equal(predictions_thresholded, 1), tf.keras.backend.floatx())

        # setting the maxim values
        labels_c_sum = tf.keras.backend.sum(labels_c)
        pred_c_sum = tf.keras.backend.sum(pred_c)

        # calculating IoU
        intersect = tf.keras.backend.sum(labels_c * pred_c)
        union = labels_c_sum + pred_c_sum - intersect
        iou = intersect / union

        return iou


class Evaluate:
    """
    This class evaluates an prediction against a evaluation dataset, stored in '/DeepLearning/evaluate_annotations/',
    under a given evaluation name folder. The eval_name is used to keep control over different sets of evaluation.
    (ex. project/DeepLearning/evaluation_annotations/test_evaluation/true_objects.shp)
    For this evaluation, numbers of true, false, missing and not labeled detections are produced. To help understand the
    quality of the prediction.

    Parameters
    ----------
    project_folder : path to the project folder ex. 'C:/project/'

    project_area : name of the project area ex. 'Naljanka'

    true_objects : name of shapefile (.shp) with true objects for validation.

    false_objects : name of shapefile (.shp) with false objects for validation. (can be None)

    eval_aoi : name of shapefile (.shp) for area where the evaluation is done.
                    If None, bounding box of the evaluation shapes will be used

    model_timestamp : the timestamp for the model used for prediction.

    eval_name : a name used to describe the evaluation

    """

    def __init__(self, project_folder, true_objects, false_objects, pred_type, eval_aoi, model_timestamp, eval_name):
        self.project_folder = project_folder
        self.true_objects = true_objects
        self.false_objects = false_objects
        self.pred_type = pred_type
        self.eval_aoi = eval_aoi
        self.model_timestamp = model_timestamp
        self.eval_name = eval_name

        # these folders is where the script looks for the needed data
        self.model_folder = self.project_folder + '/DeepLearning/model/'
        self.eval_folder = self.project_folder + '/DeepLearning/evaluate_annotations/'

        # these folders will be generated automatically
        self.report_folder = self.project_folder + '/DeepLearning/reports/'

        # these parameters can be changed to finetune the evaluation
        self.size_cut = 1  # removes objects smaller than (m2)
        self.prob_cut = 0.8  # removes objects with smaller probability than

        # cuts out the predictions to fit the evaluation area
        self.label_objects, self.predict_objects = self.get_object_data()

        # generates number for detection quality
        self.stats = self.generate_statistics()

        # creates a report based on numbers generated
        self.report = self.generate_report()

        # saves report to .csv
        self.save_report()

    def get_object_data(self):
        ''' gather testing labels, filters and cuts prediction to the same area '''
        label_objects = gp.GeoDataFrame()

        true_df = gp.read_file(self.eval_folder + self.eval_name + '/' + self.true_objects)
        true_df['label'] = 1
        label_objects = label_objects.append(true_df[['label', 'geometry']])

        if self.false_objects:
            false_df = gp.read_file(self.eval_folder + self.eval_name + '/' + self.false_objects)
            false_df['label'] = 2
            label_objects = label_objects.append(false_df[['label', 'geometry']])

        if self.eval_aoi:
            aoi = gp.read_file(self.eval_folder + self.eval_name + '/' + self.eval_aoi)
        else:
            label_bounds = label_objects.bounds
            bbox = box(label_bounds['minx'].min(), label_bounds['miny'].min(),
                       label_bounds['maxx'].max(), label_bounds['maxy'].max())
            aoi = gp.GeoDataFrame(geometry=[bbox])

        prediction_files = glob.glob(self.model_folder + '/' + self.model_timestamp +
                                     '/predictions/*/merged_shapes_' + self.pred_type + '.shp')
        predict_objects = gp.GeoDataFrame()
        for prediction_file in prediction_files:
            predict_objects = predict_objects.append(gp.read_file(prediction_file))
        predict_objects = gp.clip(predict_objects, aoi)
        predict_objects = predict_objects.loc[predict_objects.geometry.area >= self.size_cut]
        predict_objects = predict_objects.loc[predict_objects['prob'] >= self.prob_cut]

        return label_objects, predict_objects

    def generate_statistics(self):
        ''' using overlap analysis to check the quality of the prediction against the labels '''

        predict_objects = self.predict_objects
        label_objects = self.label_objects
        predict_objects['geometry'] = predict_objects.geometry.buffer(0)
        label_objects['geometry'] = label_objects.geometry.buffer(0)
        true_objects = label_objects.loc[label_objects['label'] == 1]
        false_objects = label_objects.loc[label_objects['label'] == 2]
        predict_objects = predict_objects.reset_index(drop=True)
        true_objects = true_objects.reset_index(drop=True)
        false_objects = false_objects.reset_index(drop=True)
        predict_objects['d_id'] = predict_objects.index
        true_objects['t_id'] = true_objects.index
        false_objects['f_id'] = false_objects.index
        # find prediction amount
        num_predicted = len(predict_objects)
        # find true detections
        true_detected = gp.overlay(true_objects, predict_objects, how='intersection')
        true_detected.index = true_detected['t_id']
        true_detected['d_overlap'] = true_detected.geometry.area / true_objects.geometry.area
        num_full_detected = len(true_detected.loc[true_detected['d_overlap'] >= 0.9])
        num_partly_detected = len(true_detected.loc[true_detected['d_overlap'] < 0.9])
        # find not detected
        num_not_detected = len(true_objects) - (num_full_detected + num_partly_detected)
        # find not labeled detections
        predict_rest = gp.overlay(predict_objects, label_objects, how='difference')
        predict_rest.index = predict_rest['d_id']
        predict_objects['l_overlap'] = 1 - predict_rest.geometry.area / predict_objects.geometry.area
        num_not_labeled = len(predict_objects.loc[predict_objects['l_overlap'] == 0])
        # find confirmed false detections
        num_false_detected = len(gp.overlay(false_objects, predict_objects, how='intersection'))

        return num_predicted, num_full_detected, num_partly_detected, num_not_detected, num_not_labeled, num_false_detected

    def generate_report(self):
        ''' Generates a reprot for the numbers calculated '''

        report_df = pd.DataFrame(index=['Predicted','Fully detected', 'Partly detected',
                                        'Not detected', 'Detected with no label', 'False detected'],
                                 columns=['Amount'])
        report_df.loc['Predicted','Amount'] = self.stats[0]
        report_df.loc['Fully detected', 'Amount'] = self.stats[1]
        report_df.loc['Partly detected', 'Amount'] = self.stats[2]
        report_df.loc['Not detected', 'Amount'] = self.stats[3]
        report_df.loc['Detected with no label', 'Amount'] = self.stats[4]
        report_df.loc['False detected', 'Amount'] = self.stats[5]

        return report_df

    def save_report(self):
        output_dir = self.report_folder + self.model_timestamp + '/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.report.to_csv(output_dir + self.eval_name + '.csv')


project_folder = 'C:/project/'


"""
Examples of basic usage

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

Training(project_folder=project_folder,
         true_objects='Tar_kilns_update_1.shp',
         false_objects='OtherFeatures_update_1.shp',
         feature_list=['02_TPI/radius_20/','02_TPI/radius_30/'],
         test_mapsheets=['##'],
         fetch_data=False,
         data_timestamp='1221164907',
         augment=True,
         train_on_data=True)

Predict(project_folder=project_folder,
        project_areas=['Naljanka','Kuivaniemi'],
        feature_list=['02_TPI/radius_20/','02_TPI/radius_30/'],
        model_timestamp='20211222-105650',
        test_mapsheets=['##'],
        pred_type='all')

Evaluate(project_folder=project_folder,
         true_objects='evaluate_true.shp',
         false_objects='evaluate_false.shp',
         pred_type='all',
         eval_aoi=None,
         model_timestamp='20211222-105650',
         eval_name='test1')

"""
