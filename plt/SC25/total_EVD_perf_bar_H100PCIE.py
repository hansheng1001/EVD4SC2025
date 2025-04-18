import matplotlib.pyplot as plt
import numpy as np

labels = ['4096', '8192', '12288', '16384', '20480', '24576', '28672', '32768', '36864', '40960', '45056', '49152']
# cuSOLVERSVD = np.array([2422.405273, 12012.124023, 33887.925781, 71930.656250, 135340.140625, 0])/1000

MAGMA_BRD = np.array([0.09, 0.28, 0.57, 1.10, 1.72, 2.81, 4.15, 6.06, 8.35, 11.80, 15.00, 31.30])
MAGMA_BC_VECTOR = np.array([0.19, 0.35, 0.70, 1.16, 1.91, 3.00, 4.36, 6.28, 8.86, 11.03, 14.15, 18.42])
MAGMA_DC = np.array([0.80, 1.69, 3.44, 4.10, 7.57, 7.96, 9.38, 10.57, 17.73, 19.98, 22.32, 23.52])
MAGMA_BC_BACK = np.array([0.21, 0.73, 1.81, 3.73, 6.25, 10.22, 14.88, 22.12, 31.02, 48.61, 59.70, 79.34])
MAGMA_SBR_BACK = np.array([0.03, 0.18, 0.55, 1.26, 2.50, 4.10, 6.48, 9.45, 13.10, 18.81, 21.64, 27.78])

MAGMA_BC_NO_VECTOR = np.array([0.07, 0.13, 0.26, 0.39, 0.57, 0.77, 1.01, 1.33, 1.96, 2.00, 2.44, 2.88])


MAGMA_BC_BACK_PRE = MAGMA_BC_VECTOR - MAGMA_BC_NO_VECTOR
MAGMA_REAL_BCBACK = MAGMA_BC_BACK + MAGMA_BC_BACK_PRE
# print(MAGMA_REAL_BCBACK)


MAGMA_REAL_BACK = MAGMA_DC + MAGMA_REAL_BCBACK + MAGMA_SBR_BACK

MAGMA_EVD = MAGMA_BRD + MAGMA_BC_VECTOR + MAGMA_DC + MAGMA_BC_BACK + MAGMA_SBR_BACK


WangSy2tr = np.array([207.710388, 84.1747, 768.237427, 1224.333252, 1885.714844, 2731.167236, 3853.04834, 5226.492676, 7008.606445, 10005.95117, 11612.04297, 14504.61035])/1000



ourGEMM = np.array([19.5977, 443.907227, 222.813, 461.335, 823.746, 1345.52, 2093.47, 3001.86, 4193.48, 5636.27, 7297.98, 16937.8])/1000
ourSBRBack = np.array([9.14743, 61.3239, 199.545, 465.447, 930.924, 1625.04, 2601.11, 3851.93, 5505.91, 7682.98, 10025.9541, 13115.9043])/1000
ourBCBack = np.array([25.678976, 184.062073, 596.4646, 1368.47644, 2660.593506, 4517.317383, 7189.095215, 10681.02344, 15297.60645, 21008.18164, 27978.59375, 36328.00781])/1000

MKLDC = np.array([239.233, 704.496, 1262.14, 2558.59, 3834.58, 6334.15, 9558.02, 13868.1, 16902.5, 23629.3, 29324.9, 40639])/1000

# ourBack = np.max(ourSBRBack+ourBCBack, MKLDC)
ourBack = np.maximum(ourSBRBack + ourBCBack, MKLDC)
# print(ourBack)
ourRealBack = ourBack + ourGEMM

ourEVD = WangSy2tr + ourRealBack
# print('ourEVD:', ourEVD)
# print(ourBack)

WangBack = MAGMA_DC + MAGMA_REAL_BCBACK + ourSBRBack
wangEVD = WangSy2tr + WangBack
# print('wangEVD:', wangEVD)


# cuSOLVER_EVD_VECTOR = np.array([924.781128, 5556.293945, 17071.15625, 39355.007812, 76633.914062, 126656.625])/1000
cuSOLVER_EVD_VECTOR = np.array([196.35437, 1183.612183, 3475.287354, 7527.314941, 14271.93262, 23711.60156, 37189.32422, 54696.26563, 77396.16406, 109951.2969, 139749.0625, 178331])/1000



MAGMA_BACK = MAGMA_DC + MAGMA_BC_BACK + MAGMA_SBR_BACK


MAGMA_EVDx = MAGMA_EVD / ourEVD
cuSOLVER_EVDx = cuSOLVER_EVD_VECTOR / ourEVD
wangEVDx = wangEVD / ourEVD
print('MAGMA_EVDx:', MAGMA_EVDx)
print('cuSOLVER_EVDx:', cuSOLVER_EVDx)
print('wangEVDx:', wangEVDx)

MAGMA_EVDx_avg = np.mean(MAGMA_EVDx)
MAGMA_EVDx_max = np.max(MAGMA_EVDx)
print(f'MAGMA_EVDx_avg: {MAGMA_EVDx_avg:.2f}')
print(f'MAGMA_EVDx_max: {MAGMA_EVDx_max:.2f}')


indices = [i for i, label in enumerate(labels) if int(label) > 8192]

cuSOLVER_EVDx_avg = np.mean(cuSOLVER_EVDx[indices])
cuSOLVER_EVDx_max = np.max(cuSOLVER_EVDx[indices])
# cuSOLVER_EVDx_avg = np.mean(cuSOLVER_EVDx)
# cuSOLVER_EVDx_max = np.max(cuSOLVER_EVDx)
print(f'cuSOLVER_EVDx_avg: {cuSOLVER_EVDx_avg:.2f}')
print(f'cuSOLVER_EVDx_max: {cuSOLVER_EVDx_max:.2f}')

wangEVDx_avg = np.mean(wangEVDx)
wangEVDx_max = np.max(wangEVDx)
print(f'wangEVDx_avg: {wangEVDx_avg:.2f}')
print(f'wangEVDx_max: {wangEVDx_max:.2f}')

x = np.arange(len(labels))  
width = 0.22  

fig, ax = plt.subplots()

bottom = MAGMA_BRD + MAGMA_BC_VECTOR
rects1 = ax.bar(x - 1.5*width, bottom, width, color='#CC0000', label='MAGMA SYTRD')
# rects2 = ax.bar(x - 0.5*width, MAGMA_BC_VECTOR, width, color='#FF6F61', bottom=MAGMA_DC, label='MAGMABC')
rects7 = ax.bar(x - 1.5*width, MAGMA_BACK, width, color='#FF33FF', bottom=bottom, label='MAGMA BACK')


rects8 = ax.bar(x-0.5*width, cuSOLVER_EVD_VECTOR, width, color='#B5739D', label='cuSOLVER EVD')

rects9 = ax.bar(x + 0.5*width, WangSy2tr, width, color='#FF6F61', label='Wang\'s SYTRD')
# rects2 = ax.bar(x - 0.5*width, MAGMA_BC_VECTOR, width, color='#FF6F61', bottom=MAGMA_DC, label='MAGMABC')
rects10 = ax.bar(x + 0.5*width, WangBack, width, color='#D58BFF', bottom=WangSy2tr, label='Wang\'s BACK')


rects3 = ax.bar(x + 1.5*width, WangSy2tr, width, color='#FF6F61')
rects4 = ax.bar(x + 1.5*width, ourBack, width, color='green', bottom=WangSy2tr, label='Proposed BACK')
bottom = WangSy2tr + ourBack
rects5 = ax.bar(x + 1.5*width, ourGEMM, width, color='pink', bottom=bottom, label='CUDA GEMM')


ax.set_xlabel('Matrix Size (nxn)', fontsize=14)
ax.set_ylabel('Elapsed time (s)', fontsize=14)
ax.set_xticks(x)
# ax.set_xticklabels(labels)
ax.set_xticklabels(labels, fontsize=9)


ax.legend()



for i in range(0, 6):
    ax.annotate(f'{MAGMA_EVDx[i]:.2f}x', xy=(rects1[i].get_x() + rects1[i].get_width() / 2, MAGMA_EVD[i]), textcoords="offset points",xytext=(-4, 2), ha='center', va='bottom', fontsize=7)
    ax.annotate(f'{cuSOLVER_EVDx[i]:.2f}x', xy=(rects8[i].get_x() + rects8[i].get_width() / 2, cuSOLVER_EVD_VECTOR[i]), textcoords="offset points",xytext=(0, -5), ha='center', va='bottom', fontsize=7)
    ax.annotate(f'{wangEVDx[i]:.2f}x', xy=(rects10[i].get_x() + rects10[i].get_width() / 2, wangEVD[i]), textcoords="offset points",xytext=(4, -2), ha='center', va='bottom', fontsize=7)


for i in range(6, len(x)):
    ax.annotate(f'{MAGMA_EVDx[i]:.2f}x', xy=(rects1[i].get_x() + rects1[i].get_width() / 2, MAGMA_EVD[i]), textcoords="offset points",xytext=(-6, 2), ha='center', va='bottom', fontsize=7)
    ax.annotate(f'{cuSOLVER_EVDx[i]:.2f}x', xy=(rects8[i].get_x() + rects8[i].get_width() / 2, cuSOLVER_EVD_VECTOR[i]), textcoords="offset points",xytext=(6, -2), ha='center', va='bottom', fontsize=7)
    ax.annotate(f'{wangEVDx[i]:.2f}x', xy=(rects10[i].get_x() + rects10[i].get_width() / 2, wangEVD[i]), textcoords="offset points",xytext=(6, -2), ha='center', va='bottom', fontsize=7)


mid_index = (labels.index('24576') + labels.index('28672')) / 2  
mid_position = x[int(mid_index)] + 2*width + 1/4*width  

ax.axvline(x=mid_position, color='red', linestyle='--', linewidth=1.5) 


fig.tight_layout()

plt.savefig('TotalEVD_perf_bar_H100PCIE_49152.png')
plt.savefig('TotalEVD_perf_bar_H100PCIE_49152.pdf')

plt.show()
