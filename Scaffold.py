import astra
import numpy as np
import flexraytools as flex
from matplotlib import pyplot as plt
import time
import cv2
import glob

start = time.time()
angle_skip     = 4 # 1 means no skip
binning        = 4 # 1 means no binning
iterations     = 20
n_proj_per_rot = 2880
corrected_COR = 989.61
# data
path = '/home/anhtuan299/Documents/Datasets/Diamond/Projection/'
proj_geom = flex.geometry_parsing.parse_to_astra_geometry(path + "Acquisition settings XRE.txt",
                                                          binning=binning,
                                                          proj_skip=angle_skip,
                                                          COR=corrected_COR,
                                                          verbose=True)
# vectors = proj_geom["Vectors"]
# pixel_size = np.linalg.norm(vectors[0, 6:9])
# vectors[:, 2] = np.linspace(0, 4.631*4/binning, vectors.shape[0])
# vectors[:, 5] = np.linspace(0, 4.631*4/binning, vectors.shape[0])

def linspline(x, n_samples):
    return np.interp(np.arange(n_samples), np.arange(len(x))/(len(x)-1)*(n_samples-1), x)
x = np.array([[-0.70053877,  2.03604759,  2.29478889,  0.04515735,  0.52966854],
              [ 1.31391857,  1.88584762, -1.56157154, -0.7879991 , -0.87279179],
              [-2.49571218, -1.44555423, -0.63424357, -0.06457296,  1.65762124],
              [ 0.08420747, -0.04316858, -0.01389955, -0.02236995,  0.00441405],
              [-0.11861882, -0.12128123, -0.08365052,  0.05831938, -0.02361169],
              [ 0.05867118, -0.1014062 , -0.03507089, -0.04522655, -0.13715424]])
# x = np.array([[-1.40107754e+00,  4.07209518e+00,  4.58957778e+00,  9.03147000e-02,  1.05933708e+00],
#               [ 2.62783714e+00,  3.77169524e+00, -3.12314308e+00, -1.57599820e+00, -1.74558358e+00],
#               [-4.99142436e+00, -2.89110846e+00, -1.26848714e+00, -1.29145920e-01,  3.31524248e+00],
#               [ 8.42074700e-02, -4.31685800e-02, -1.38995500e-02, -2.23699500e-02,  4.41405000e-03],
#               [-1.18618820e-01, -1.21281230e-01, -8.36505200e-02,  5.83193800e-02, -2.36116900e-02],
#               [ 5.86711800e-02, -1.01406200e-01, -3.50708900e-02, -4.52265500e-02, -1.37154240e-01]])
x = x.reshape((6, -1))

vectors = proj_geom["Vectors"]
x_shift = linspline(x[0], vectors.shape[0])
y_shift = linspline(x[1], vectors.shape[0])
z_shift = linspline(x[2], vectors.shape[0])
x_rot = linspline(x[3], vectors.shape[0])/10
y_rot = linspline(x[4], vectors.shape[0])/10
z_rot = linspline(x[5], vectors.shape[0])/10
R_x = lambda t: np.array([[1,         0,          0],
                          [0, np.cos(t), -np.sin(t)],
                          [0, np.sin(t),  np.cos(t)]])
R_y = lambda t: np.array([[ np.cos(t), 0, np.sin(t)],
                          [         0, 1,         0],
                          [-np.sin(t), 0, np.cos(t)]])
R_z = lambda t: np.array([[np.cos(t), -np.sin(t), 0],
                          [np.sin(t),  np.cos(t), 0],
                          [        0,          0, 1]])
for i in range(vectors.shape[0]):
    Q = R_x(x_rot[i]) @ R_y(y_rot[i]) @ R_z(z_rot[i])
    b = np.array([x_shift[i], y_shift[i], z_shift[i]])
    vec = vectors[i]
    vec[:3] = Q @ vec[:3]+ b
    vec[3:6] = Q @ vec[3:6] + b
    vec[6:9] = Q @ vec[6:9]
    vec[9:] = Q @ vec[9:]

flex_geom           = flex.geometry_parsing.parse_acquisition_settings(path + "Acquisition settings XRE.txt")
flex_geom["angles"] = np.linspace(0, 2*np.pi, n_proj_per_rot)
print("Reading data")

num_proj, num_flat, num_dark = flex.parse_field(path + "Acquisition settings XRE.txt", ['total projections',
                                                                                        'number gain images',
                                                                                        'number offset images'], int)
data = flex.load_data(path + "scan_000000.tif", range(0, n_proj_per_rot, angle_skip), binning, crop_box=None)
flat = flex.load_data(path + "io000000.tif", num_flat, binning,  crop_box=None)
dark = flex.load_data(path + "di000000.tif", num_dark, binning,  crop_box=None)

print("flat field correction and log normalization")
data = flex.log_correct(data, flat, dark, astra_order=True)

print("""
+----------------+
| Reconstruction |
+----------------+
""")

# create volume geometry
rec_shape = (data.shape[2], data.shape[2], data.shape[0])
vol_geom  = astra.create_vol_geom(rec_shape)

# create astra objects for volume and projection data
rec_id   = astra.data3d.create("-vol", vol_geom)
sino_id  = astra.data3d.create("-sino", proj_geom, data)

print("reconstructing")

plt.ion() # interactive mode such that python keeps running when showing plots
fig, ax = plt.subplots() # on this figure we will show the currenct reconstruction each iteration
plt.pause(0.01)

def callback(iteration, current_solution=None, *args, **kwargs):
    shape = (vol_geom["GridSliceCount"], vol_geom["GridColCount"], vol_geom["GridRowCount"])
    ax.imshow(current_solution.reshape(shape)[:, :, shape[2]//2], cmap="gray")
    ax.set_title("Iteration " + str(iteration))
    plt.pause(0.01)

print('----------Reconstructing with FleXrayTools----------')
print('using the Barzilai-Borwein Method')
x_rec_BB = flex.BB(sino_id, rec_id, bounds=(0, np.inf), iterations=iterations, verbose=True, callback=callback)
# x_rec_CGLS = flex.CGLS(sino_id, rec_id, iterations=iterations, verbose=True, callback=callback)
plt.ioff() # turn off interactive mode

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(x_rec_BB[x_rec_BB.shape[0]//2, :, :], cmap="gray")
plt.title('central slice along Z-axis')
plt.subplot(1, 3, 2)
plt.imshow(x_rec_BB[:, x_rec_BB.shape[1]//2, :], cmap="gray")
plt.title('central slice along Y-axis')
plt.subplot(1, 3, 3)
plt.imshow(x_rec_BB[:, :, x_rec_BB.shape[2]//2], cmap="gray")
plt.title('central slice along X-axis')
plt.show()

finish = time.time()

print('running time =', finish-start, '(second)')
ans = input("Save this? (Y/n): ")
if ans != "n":
    name = input("name?: ")
    print("saving result")
    flex.save_data(x_rec_BB, '/home/anhtuan299/Documents/Datasets/Diamond/Reconstruction0605/' + name, file_type='.tif')