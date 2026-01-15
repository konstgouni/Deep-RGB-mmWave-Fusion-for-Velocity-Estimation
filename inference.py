from ast import literal_eval
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, calcOpticalFlowFarneback, COLOR_BGR2RGB, resize
from datetime import datetime
from logging import ERROR
from numpy import loadtxt, int32, float32, arange, tan, deg2rad, arctan2, zeros, array, meshgrid, mean as npmean, empty, expand_dims, absolute, concatenate
from numpy.linalg import lstsq, solve
from os import environ, listdir
from os.path import dirname, abspath, join, exists, splitext, basename
from pandas import read_csv, DataFrame
from pyproj import CRS, Transformer
from random import seed, shuffle
from shutil import rmtree
from time import perf_counter
from warnings import simplefilter

N_BEAMS, nSamples, eps, fov_diag, orig_width, orig_height = 64, 5964, 1e-5, 110.0, 960.0, 540.0

def tfGPUProperImportNoWarnings(logLevelString):
    global device
    environ['TF_CPP_MIN_LOG_LEVEL'] = logLevelString
    simplefilter(action = 'ignore', category = FutureWarning)
    simplefilter(action = 'ignore', category = Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(ERROR)
    gpus = tf.config.list_physical_devices("GPU")
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print('Tensorflow will use the GPU')
        device = '/GPU:0'
    except Exception:
        print('Tensorflow will use the CPU')
        device = '/CPU:0'
    return tf

def tfCPUProperImportNoWarnings(logLevelString):
    global device
    environ['TF_CPP_MIN_LOG_LEVEL'] = logLevelString
    simplefilter(action = 'ignore', category = FutureWarning)
    simplefilter(action = 'ignore', category = Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(ERROR)
    cpus = tf.config.list_physical_devices("CPU")
    tf.config.set_visible_devices(cpus[0], "CPU")
    print('Tensorflow will use Intel CORE i7 CPU')
    device = '/CPU:0'
    return tf

tf = tfCPUProperImportNoWarnings('3') # tfGPUProperImportNoWarnings('3') 
mmWaveModel = tf.keras.models.load_model("./mmWave_Polar_Localizer.keras", compile = False)
sm = tf.saved_model.load('./Adam3_FineTunedRGB_mmWave_Velocity_ModelF_savedmodel')
infer = sm.signatures["serving_default"]

def publishablePlotFonts():
    from matplotlib import pyplot as plt
    plt.rc('font', size = 10)                       
    plt.rc('axes', titlesize = 12)
    plt.rc('axes', labelsize = 10)
    plt.rc('xtick', labelsize = 9)
    plt.rc('ytick', labelsize = 9)
    plt.rc('legend', fontsize = 9)
    plt.rc('figure', titlesize = 12)
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']
    return plt

def deletePycacheFolder(toDel = True):
    if toDel:
        script_directory = dirname(abspath(__file__))
        pycache_folder = join(script_directory, '__pycache__')
        if exists(pycache_folder):
            rmtree(pycache_folder)
            print(f"Deleted: {pycache_folder}")
        else:
            print(f"__pycache__ folder does not exist in {script_directory}")
    else:
        print('pycache folder will not be deleted.')

def readExperimentCycleExtended(file_path):
    try:
        data = loadtxt(file_path, dtype = int32)
        cycles = [(start - 1, end - 1) for (start, end) in data]
        return cycles
    except Exception as e:
        print(f"Failed to read experiment cycle data: {e}")
        return []
    
def readAnnotatedBoundingBox(path):
    try:
        arr = loadtxt(path, dtype = float32)
        if arr is None:
            return []
        arr = arr.reshape(1, 5) if arr.ndim == 1 else arr
        tx_boxes = arr[arr[:, 0] == 0] # class_id == 0 (UE)
        if len(tx_boxes) == 0:
            return []  
        tx_boxes = sorted(tx_boxes, key = lambda b: b[3] * b[4], reverse = True) # If multiple UE boxes appear (rare), take the largest box
        _, x_c, y_c, w, h = tx_boxes[0]
        return [x_c, y_c, w, h]
    except Exception as e:
        return []

def imageDataPaths(df):
    img_rel_paths = df['unit1_rgb'].values
    return img_rel_paths

def gpsDataPaths(df, dataDir):
    baseStationPosRelPath = df['unit1_loc'].values[0]
    bsPos = loadtxt(join(dataDir, baseStationPosRelPath))
    uePosRelPath = df['unit2_loc_cal'].values
    return bsPos, uePosRelPath

def mmWaveDataPaths(df):
    powersRelPath = df['unit1_pwr_60ghz']
    beam_idxs = arange(N_BEAMS) + 1
    return beam_idxs, powersRelPath

def creategpsENUTransformation(ref_lat, ref_lon):
    enu_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={ref_lat} +lon_0={ref_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, enu_crs, always_xy = True)
    return transformer

def gpsENU(transformer, vehicle_latitude, vehicle_longitude):
    E, N = transformer.transform(vehicle_longitude, vehicle_latitude)
    return E, N

def imageDataNormalization(frame):
    return frame / 255.0

def minMaxDenorm(x_norm, x_min, x_max, eps = 1e-5):
    x = x_norm * ((x_max - x_min) + eps) + x_min
    return x

def calculateDeltaT(sample_idx, df):
    fmt = "%H-%M-%S-%f"
    timeStamp = df['time_stamp[UTC]'][sample_idx]
    timeStamp2 = df['time_stamp[UTC]'][sample_idx + 1]
    if timeStamp is None or timeStamp2 is None:
        return None
    t1_str = literal_eval(timeStamp)[0]
    t2_str = literal_eval(timeStamp2)[0]
    d1 = datetime.strptime(t1_str, fmt)
    d2 = datetime.strptime(t2_str, fmt)
    dt_true = (d2 - d1).total_seconds()
    if dt_true <= 1e-6:
        print(f'In between samples {sample_idx} and {sample_idx + 1}, dt was found to be {dt_true} s. dt will fall back to the nominal of 0.1 s')
        return 0.1
    return dt_true

def computeCameraIntrinsics(fov_diag, orig_width, orig_height):
    aspect_ratio = orig_width / orig_height  
    tan_half_diag = tan(0.5 * deg2rad(fov_diag))
    tan_half_h = tan_half_diag / ((1 + (1 / aspect_ratio) ** 2) ** 0.5)
    fov_h = 2 * arctan2(tan_half_h, 1.0)
    f = 0.5 * orig_width / tan(0.5 * fov_h)
    cu, cv = 0.5 * orig_width, 0.5 * orig_height
    return f, cu, cv

def generate_zeros(imgs_t_1, imgs_t, pwr, dt_true_vector, vc, bbox_t_1):
    imgs_t_1.append(zeros((240, 320, 3), dtype = float32))
    imgs_t.append(zeros((240, 320, 3), dtype = float32))
    pwr.append(zeros((N_BEAMS, ), dtype = float32))
    dt_true_vector.append(0.0)  
    vc.append(array([0.0, 0.0], dtype = float32))
    bbox_t_1.append(array([0.0, 0.0, 0.0, 0.0], dtype = float32))
  
def compute_flow_DL_fusion(dataDir, cvBBoxAnnotPath, cycle):
    global sm, infer, mmWaveModel
    csv_file = [f for f in listdir(dataDir) if f.endswith('.csv')][0]
    df = read_csv(join(dataDir, csv_file))
    imgRelPaths = imageDataPaths(df)
    bsPos, uePosRelPath = gpsDataPaths(df, dataDir)
    _, powersRelPath = mmWaveDataPaths(df)
    coordTrans = creategpsENUTransformation(bsPos[0], bsPos[1])
    cycleStart, cycleEnd = cycle
    selectedSamples = range(cycleStart, cycleEnd + 1)
    pwr, vc, dt_true_vector = [], [], []
    imgs_t_1, imgs_t, bbox_t_1 = [], [], []
    for sample_idx in selectedSamples:
        # RGB images acquisition
        img_path_t_1 = join(dataDir, imgRelPaths[sample_idx])
        img_path_t = join(dataDir, imgRelPaths[sample_idx + 1])
        frame_t_1 = imread(img_path_t_1)
        frame_t = imread(img_path_t)
        if frame_t_1 is None or frame_t is None:
            generate_zeros(imgs_t_1, imgs_t, pwr, dt_true_vector, vc, bbox_t_1)
            continue
        imgs_t_1.append(frame_t_1); imgs_t.append(frame_t)
        # bounding box bb(t-1)
        imgPrevBBoxAnnotFile = splitext(basename(imgRelPaths[sample_idx]))[0] + ".txt"
        bboxPrevPath = join(cvBBoxAnnotPath, imgPrevBBoxAnnotFile)
        currBbox = readAnnotatedBoundingBox(bboxPrevPath)
        if len(currBbox) != 4:
            generate_zeros(imgs_t_1, imgs_t, pwr, dt_true_vector, vc, bbox_t_1)
            continue
        m_x_Prev, m_y_Prev, w_Prev, h_Prev = currBbox
        x_min_Prev, y_min_Prev, w_Prev, h_Prev = int((m_x_Prev - 0.5 * w_Prev) * orig_width), int((m_y_Prev - 0.5 * h_Prev) * orig_height), int(w_Prev * orig_width), int(h_Prev * orig_height)
        bbox_t_1.append(array([x_min_Prev, y_min_Prev, w_Prev, h_Prev]))
        # p_c(t-1), p_c(t)
        posAbsPath_t_1 = join(dataDir, uePosRelPath[sample_idx])
        vehicle_lat_t_1, vehicle_long_t_1 = loadtxt(posAbsPath_t_1)
        posAbsPath_t = join(dataDir, uePosRelPath[sample_idx + 1])
        vehicle_lat_t, vehicle_long_t = loadtxt(posAbsPath_t)
        if vehicle_lat_t_1 is None or vehicle_long_t_1 is None or vehicle_lat_t is None or vehicle_long_t is None:
            generate_zeros(imgs_t_1, imgs_t, pwr, dt_true_vector, vc, bbox_t_1)
            continue
        E_t_1, N_t_1 = gpsENU(coordTrans, vehicle_lat_t_1, vehicle_long_t_1)
        pC1_SDE_t_1 = array([-1.0 * N_t_1, E_t_1])
        E_t, N_t = gpsENU(coordTrans, vehicle_lat_t, vehicle_long_t)
        pC1_SDE_t = array([-1.0 * N_t, E_t])
        # dt
        dt_true = calculateDeltaT(sample_idx, df)
        if dt_true is None:
            generate_zeros(imgs_t_1, imgs_t, pwr, dt_true_vector, vc, bbox_t_1)
            continue
        dt_true_vector.append(dt_true)
        # v_c(t)
        velocityVectorGT = (pC1_SDE_t - pC1_SDE_t_1) / dt_true
        vc.append(velocityVectorGT)
        # mmWave received powers' acquisition
        pwrs_t_1 = loadtxt(join(dataDir, powersRelPath[sample_idx]))
        if pwrs_t_1 is None:
            generate_zeros(imgs_t_1, imgs_t, pwr, dt_true_vector, vc, bbox_t_1)
            continue
        pwr.append(pwrs_t_1)
        # end of data retrieval
    vc = array(vc, dtype = float32)
    imgs_t_1 = array(imgs_t_1, dtype = float32)
    imgs_t = array(imgs_t, dtype = float32)
    pwr = array(pwr, dtype = float32)
    dt_true_vector = array(dt_true_vector, dtype = float32)
    bbox_t_1 = array(bbox_t_1, dtype = int32)
    # Time Vector Calculation for cycleStart .. cycleEnd + 1
    t_frames = zeros(len(dt_true_vector) + 1, dtype = float32)
    t_frames[1 : ] = dt_true_vector.cumsum()
    # t_step_start = t_frames[ : -1] # time at frame i
    t_step_mid = 0.5 * (t_frames[ : -1] + t_frames[1 : ]) # midpoint of (i, i + 1)
    plt = publishablePlotFonts()
    # Dense Optical Flow Tracking, Farneback Method
    t0_optical_flow = perf_counter()
    of_states = []
    inference_times_of = []
    f, cu, cv = computeCameraIntrinsics(fov_diag, orig_width, orig_height) 
    rho_phi_minValues, rho_phi_maxValues = array([9.069541, -0.7731396]), array([14.801912, 0.783113])
    lambda_I_ridge = 1e-2 * array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype = float32)
    alphas = array([0.90, 0.97], dtype = float32)
    complementary_alphas = array([0.10, 0.03], dtype = float32)
    of_filt = None
    for counter, _ in enumerate(selectedSamples):
        pwr_i = pwr[counter]
        old_frame = imgs_t_1[counter]
        new_frame = imgs_t[counter]
        if not old_frame.any() or not new_frame.any() or not pwr_i.any() or not vc[counter].any():
            of_states.append(array([float('nan'), float('nan')], dtype = float32))
            inference_times_of.append(float('nan'))
            continue
        t0_cl = perf_counter()
        old_gray = cvtColor(old_frame, COLOR_BGR2GRAY)
        new_gray = cvtColor(new_frame, COLOR_BGR2GRAY)
        flow = calcOpticalFlowFarneback(old_gray, new_gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
        x_min_Prev, y_min_Prev, w_Prev, h_Prev = bbox_t_1[counter]
        x1 = max(0, x_min_Prev)
        y1 = max(0, y_min_Prev)
        x2 = min(flow.shape[1], x_min_Prev + w_Prev)
        y2 = min(flow.shape[0], y_min_Prev + h_Prev)
        if x2 <= x1 or y2 <= y1:
            of_states.append(array([float('nan'), float('nan')], dtype = float32))
            inference_times_of.append(float('nan'))
            continue
        # --- Use all (or many) flow points inside bbox --- 
        flow_roi = flow[y1 : y2, x1 : x2, : ] 
        dt_true =  dt_true_vector[counter] # calculateDeltaT(sample_idx, df)
        mag_roi = (flow_roi[ : , : , 0] ** 2 + flow_roi[ : , : , 1] ** 2) ** 0.5 
        mask = mag_roi > 0.65 # tune threshold in pixels / frame (or remove mask completely)
        step = 1  # take every 4th pixel (tune)
        mask_sub = mask[ : : step, : : step]
        flow_sub = flow_roi[ : : step, : : step, : ]
        # absolute pixel grid for those subsampled points (SHAPE MATCHES mask_sub)
        H, W = flow_sub.shape[ : 2] 
        xs = arange(W) 
        ys = arange(H) 
        XX, YY = meshgrid(xs + x1, ys + y1) # XX, YY : (H, W) 
        u = XX[mask_sub].astype(float32) # pixel u coords 
        v = YY[mask_sub].astype(float32) # pixel v coords 
        du = (flow_sub[ : , : , 0] / dt_true)[mask_sub].astype(float32) 
        dv = (flow_sub[ : , : , 1] / dt_true)[mask_sub].astype(float32) 
        M = du.shape[0]
        if M < 6:
            # not enough points to make a stable tall system; fallback (e.g. centroid mean)
            print('Not enough tracklets')
            du_mean = npmean(flow_roi[ : , : , 0]) / dt_true
            dv_mean = npmean(flow_roi[ : , : , 1]) / dt_true
            xdot = array([du_mean, dv_mean], dtype = float32)
            currCentroid = array([(x1 + x2) / 2, (y1 + y2) / 2], dtype = float32)
            with tf.device(device):
                rho_phi_hat_n = mmWaveModel.predict({'mmWave_input': pwr_i.reshape(1, -1)}, batch_size = 32, verbose = 0).flatten()
            rho_phi_hat = minMaxDenorm(rho_phi_hat_n, rho_phi_minValues, rho_phi_maxValues, eps)
            d_hat = rho_phi_hat[0]
            x_c = currCentroid[0] - cu # pixel coords relative to principal point
            y_c = currCentroid[1] - cv
            x_n = x_c / f
            y_n = y_c / f
            inv_norm = 1.0 / ((x_n * x_n + y_n * y_n + 1.0) ** 0.5)
            Z = d_hat * inv_norm
            L = array([[f / Z, 0.0, -x_c / Z], [0.0, f / Z, -y_c / Z]], dtype = float32)
            # Least-squares / Pseudo-Inverse Min Norm solution (underdetermined system for M = 1)
            vc_optical_flow, *_ = lstsq(L, xdot, rcond = None)
        else:
            # Build stacked xdot = [du1, dv1, du2, dv2, ...]^T  shape (2M, )
            xdot = empty((2 * M, ) , dtype = float32)
            xdot[0 : : 2] = du
            xdot[1 : : 2] = dv
            # Depth from mmWave model 
            with tf.device(device):
                rho_phi_hat_n = mmWaveModel.predict({'mmWave_input': pwr_i.reshape(1, -1)}, batch_size = 32, verbose = 0).flatten()
            rho_phi_hat = minMaxDenorm(rho_phi_hat_n, rho_phi_minValues, rho_phi_maxValues, eps)
            d_hat = rho_phi_hat[0]
            # For each pixel, compute normalized ray length factor 
            x_c = (u - cu)  # pixel coords relative to principal point
            y_c = (v - cv)
            x_n = x_c / f
            y_n = y_c / f
            inv_norm = 1.0 / ((x_n * x_n + y_n * y_n + 1.0) ** 0.5)  # (M, )
            Z = d_hat * inv_norm                                     # (M, )
            # Build tall L of shape (2m, 3)
            L = zeros((2 * M, 3), dtype = float32)
            # --- translation columns [Vx, Vy, Vz]^T ---
            L[0 : : 2, 0] = f / Z
            L[0 : : 2, 2] = -x_c / Z
            L[1 : : 2, 1] = f / Z
            L[1 : : 2, 2] = -y_c / Z
            # Ridge Regression / Pseudo-Inverse solution (overdetermined system)
            A = L.T @ L + lambda_I_ridge
            b = L.T @ xdot
            vc_optical_flow = solve(A, b)
        v_meas = array([vc_optical_flow[0], vc_optical_flow[2]], dtype = float32)
        if of_filt is None:
            of_filt = v_meas
        else:
            of_filt = alphas * of_filt + complementary_alphas * v_meas
        t1_cl = perf_counter()
        dt_cl = t1_cl - t0_cl
        inference_times_of.append(dt_cl)
        of_states.append(of_filt) 
    of_states = array(of_states, dtype = float32)
    t1_optical_flow = perf_counter()
    dt_optical_flow = t1_optical_flow - t0_optical_flow
    print(f'Execution time of OF-based estimation loop was {dt_optical_flow:.3f} seconds.')
    print(f'Mean Inference Time of OF was : {npmean(inference_times_of):.3f} seconds.')
    print('End of Dense Optical Flow tracking')
    # RGB & mmWave Deep Convolutional Neural Network prediction
    t0_dl = perf_counter()
    dl_states = []
    inference_times_dl = []
    for counter, _ in enumerate(selectedSamples):
        pwr_i = pwr[counter]
        old_frame = imgs_t_1[counter]
        new_frame = imgs_t[counter]
        if not old_frame.any() or not new_frame.any() or not pwr_i.any() or not vc[counter].any():
            dl_states.append(array([float('nan'), float('nan')], dtype = float32))
            inference_times_dl.append(float('nan'))
            continue
        t0_cl = perf_counter()
        frame_t_1_rgb = cvtColor(old_frame, COLOR_BGR2RGB)
        frame_t_1_rgb_resized = resize(frame_t_1_rgb, (320, 240))
        frame_t_1_rgb_norm = imageDataNormalization(frame_t_1_rgb_resized)
        frame_t_rgb = cvtColor(new_frame, COLOR_BGR2RGB)
        frame_t_rgb_resized = resize(frame_t_rgb, (320, 240))
        frame_t_rgb_norm = imageDataNormalization(frame_t_rgb_resized)
        img_t_1 = expand_dims(frame_t_1_rgb_norm, axis = 0)
        img_t = expand_dims(frame_t_rgb_norm, axis = 0)
        with tf.device(device):
            out = infer(input_t_1 = tf.constant(img_t_1, tf.float32), input_t = tf.constant(img_t, tf.float32), mmWave_input =  tf.constant(pwr_i.reshape(1, -1), tf.float32), )
            pred = out['output_0'].numpy()[0]
        t1_cl = perf_counter()
        dt_cl = t1_cl - t0_cl
        inference_times_dl.append(dt_cl)
        dl_states.append(pred)
    dl_states = array(dl_states, dtype = float32)
    t1_dl = perf_counter()
    dt_dl = t1_dl - t0_dl
    print(f'Execution time of DL-based estimation loop was {dt_dl:.3f} seconds.')
    print(f'Mean Inference Time of DL was : {npmean(inference_times_dl):.3f} seconds.')
    print(f'Deep Fusion Model trained on DeepSense 6G Scenario 9 Dataset constituded of {nSamples} performed inference. The comparative evaluation plots will be generated.')
    # csv ground truth data generation
    data_gt_vel = {'t_s_gt': t_step_mid, 'vx_m_s_gt': vc[ : , 0], 'vz_m_s_gt': vc[ : , 1]}
    df_gt_vel = DataFrame(data_gt_vel)
    filename_gt_vel = "Ground_Truth.csv"
    df_gt_vel.to_csv(filename_gt_vel, index = False)
    print(f'Ground Truth Data has been written to {filename_gt_vel}')
    # Plots
    plt.figure(1, figsize = (7, 5), dpi = 300)
    plt.ylim(0, 8)
    plt.plot(t_step_mid, vc[ : , 0], 'g', label = 'ground truth')
    plt.plot(t_step_mid, of_states[ : , 0], 'm', label = 'dense optical flow')
    plt.plot(t_step_mid, dl_states[ : , 0], 'b', label = 'deep fusion model')
    plt.xlabel('Time t (seconds)')
    plt.ylabel('Vx (meters/second)')
    plt.title('Vehicle (User Equipment) Velocity in Camera X Axis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.14)
    plt.show()
    plt.figure(2, figsize = (7, 5), dpi = 300)
    plt.ylim(-4.5, 4.5)
    plt.plot(t_step_mid, vc[ : , 1], 'g', label = 'ground truth')
    plt.plot(t_step_mid, of_states[ : , 1], 'm', label = 'dense optical flow')
    plt.plot(t_step_mid, dl_states[ : , 1], 'b', label = 'deep fusion model')
    plt.xlabel('Time t (seconds)')
    plt.ylabel('Vz (meters/second)')
    plt.title('Vehicle (User Equipment) Velocity in Camera Z axis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.14)
    plt.show()
    MeanRelativeError_Vx = npmean(absolute(dl_states[ : , 0] - vc[ : , 0]) / absolute(vc[ : , 0] + eps))
    MeanRelativeError_Vx *= 100
    print(f'Mean Relative Error (%) for the Vx prediction of this study is : {MeanRelativeError_Vx:.3f}')
    return vc, of_states, dl_states

def main():
    dataDir, cycles = './scenario9_dev', readExperimentCycleExtended('./annotatedCycles.txt')
    cvBBoxAnnotPath = './scenario9_dev/resources/annotations/bbox'
    if len(cycles) == 0:
         print('No relevant experiments could be found.')
         deletePycacheFolder(True)
         return 
    seed(42)
    shuffle(cycles)
    trainValSetLen = round(0.850 * len(cycles)) # 65 % - 20 % - 15 % train - val - test split
    test_cycles = cycles[trainValSetLen  : ] # test_cycles = cycles
    selectedCycleIdxStr = input('Please specify the experimental cycle you would like to load (Type 2 to replicate the 4 consequent driving experiments): \n')
    selectedCycle = int(selectedCycleIdxStr) - 1
    nextCycle = selectedCycle + 1
    nextNextCycle = nextCycle + 1
    nextNextNextCycle = nextNextCycle + 1
    vc1, of_states1, dl_states1 = compute_flow_DL_fusion(dataDir, cvBBoxAnnotPath, test_cycles[selectedCycle]) 
    vc2, of_states2, dl_states2 = compute_flow_DL_fusion(dataDir, cvBBoxAnnotPath, test_cycles[nextCycle]) 
    vc3, of_states3, dl_states3 = compute_flow_DL_fusion(dataDir, cvBBoxAnnotPath, test_cycles[nextNextCycle]) 
    vc4, of_states4, dl_states4 = compute_flow_DL_fusion(dataDir, cvBBoxAnnotPath, test_cycles[nextNextNextCycle]) 
    vc = concatenate([vc1, vc2, vc3, vc4], axis = 0)
    of_states = concatenate([of_states1, of_states2, of_states3, of_states4], axis = 0)
    dl_states = concatenate([dl_states1, dl_states2, dl_states3, dl_states4], axis = 0)
    MSE_DL = npmean((dl_states - vc) ** 2)
    RMSE_DL = MSE_DL ** 0.5
    RelativeErrorVector_DL = absolute(dl_states - vc) / absolute(vc)
    print(f'DL Model Error Metrics on Test Data : MSE (m^2/s^2), RMSE (m/s), MRE_Vx (0-1) : {MSE_DL}, {RMSE_DL}, {npmean(RelativeErrorVector_DL[ : , 0])}')
    MSE_OF = npmean((of_states - vc) ** 2)
    RMSE_OF = MSE_OF ** 0.5
    RelativeErrorVector_OF = absolute(of_states - vc) / absolute(vc)
    print(f'OF Model Error Metrics on Test Data : MSE (m^2/s^2), RMSE (m/s), MRE_Vx (0-1) : {MSE_OF}, {RMSE_OF}, {npmean(RelativeErrorVector_OF[ : , 0])}')
    plt = publishablePlotFonts()
    plt.figure(3, figsize = (7, 5), dpi = 300)
    plt.ylim(0, 10)
    plt.plot(vc[ : , 0], 'g', label = 'ground truth')
    plt.plot(of_states[ : , 0], 'm', label = 'dense optical flow')
    plt.plot(dl_states[ : , 0], 'b', label = 'deep fusion model')
    plt.xlabel('Sample (k)')
    plt.ylabel('Vx (meters/second)')
    plt.title('Vehicle (User Equipment) Velocity in Camera X axis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.14)
    plt.show()
    plt.figure(4, figsize = (7, 5), dpi = 300)
    plt.ylim(-7, 7)
    plt.plot(vc[ : , 1], 'g', label = 'ground truth')
    plt.plot(of_states[ : , 1], 'm', label = 'dense optical flow')
    plt.plot(dl_states[ : , 1], 'b', label = 'deep fusion model')
    plt.xlabel('Sample (k)')
    plt.ylabel('Vz (meters/second)')
    plt.title('Vehicle (User Equipment) Velocity in Camera Z axis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.14)
    plt.show()
    deletePycacheFolder(True)

if __name__ == "__main__":
    main()
