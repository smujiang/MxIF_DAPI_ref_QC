import os
import pickle
import numpy as np
import math
from scipy import stats
import statistics
import matplotlib.pyplot as plt


case_id = "OVCA_TMA22"  # case ID
# output = "/research/bsi/projects/staff_analysis/MyLANID/MxIF_CellSeg/OME_TIFF/QC_out"
output = "H:\\MxIF_Study\\QualityChecking\\QC_Reports"
QC_out_dir = os.path.join(output, case_id, "DAPI_QC")  # output directory: save pickle files for arrays, variables.

ROI_range = range(1, 348)
fit_errs = []
for r in ROI_range:  # regions range
    print("print processing %d" % r)
    fn = os.path.join(QC_out_dir, "dapi_ssim_array", "ssim_array_region" + str(r) + ".pickle")
    fp = open(fn, 'rb')
    ssim_array = pickle.load(fp)
    m = ssim_array.shape[0]
    idx = (np.arange(1, m + 1) + (m + 1) * np.arange(m - 1)[:, None]).reshape(m, -1)
    ssim_array = ssim_array.ravel()[idx]

    vec_DAPI_SSIM_avg = np.mean(ssim_array, axis=1)
    fit_err = np.sum(1 - vec_DAPI_SSIM_avg)
    fit_errs.append(fit_err)
print(len(fit_errs))

# mean, standard diviation, median, minimum and maximum

print(statistics.mean(fit_errs))
print(statistics.stdev(fit_errs))
print(statistics.median(fit_errs))
min_val = min(fit_errs)
max_val = max(fit_errs)
print(min_val)
print(max_val)

kde = stats.gaussian_kde(fit_errs)
xx = np.linspace(min_val, max_val, 300)
n_bins = 20
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(fit_errs, density=True, bins=n_bins, alpha=0.3, edgecolor='black')
ax.plot(xx, kde(xx))
plt.grid()
plt.title("Distribution of SSIM accumulated error")
plt.show()


std_img_avg_list = []
for r in ROI_range:
    print("print processing %d" % r)
    fn = os.path.join(QC_out_dir, "dapi_std_img", str(r) + ".pickle")
    f = open(fn, 'rb')
    std_img = pickle.load(f)
    std_img_avg_list.append(np.average(std_img))
print(len(std_img_avg_list))


print(statistics.mean(std_img_avg_list))
print(statistics.stdev(std_img_avg_list))
print(statistics.median(std_img_avg_list))
min_val = min(std_img_avg_list)
max_val = max(std_img_avg_list)
print(min_val)
print(max_val)


kde = stats.gaussian_kde(std_img_avg_list)
xx = np.linspace(min_val, max_val, 300)
n_bins = 20
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(std_img_avg_list, density=True, bins=n_bins, alpha=0.3, edgecolor='black')
ax.plot(xx, kde(xx))
plt.grid()
plt.title("Distribution of DAPI std image average")
plt.show()
