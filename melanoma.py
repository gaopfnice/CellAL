import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
from itertools import product as product
import time
import psutil





nan = 0
path = 'dataset/human/'
df1 = pd.read_csv(path + 'LRI_name_known.csv', header=None, index_col=None, sep=' ').to_numpy()
df2 = pd.read_csv(path + '1-proba_name.csv', header=None, index_col=None, sep=' ').to_numpy()
ll = pd.read_csv(path + 'ligand_gen.csv', header=None, index_col=None).to_numpy()
rl = pd.read_csv(path + 'receptor_gen.csv', header=None, index_col=None).to_numpy()

df = np.vstack((df1, df2))
ligand = df[:, 0]
receptor = df[:, 1]
l_dict = {}
r_dict = {}
for i in range(ll.shape[0]):
    l_dict[ll[i, 2]] = ll[i, 1]
for i in range(rl.shape[0]):
    r_dict[rl[i, 2]] = rl[i, 0]

l_gene = []
r_gene = []
for i in ligand:
    l_gene.append(l_dict[i])
for i in receptor:
    r_gene.append(r_dict[i])

LRI_gene = np.vstack((np.array(l_gene).T, np.array(r_gene).T)).T

dt = pd.read_csv('dataset/human/GSE72056.csv', index_col=0, header=None)
# a = np.array(dt.loc["Cell"])
dict = {}
dict["Cell"] = np.array(dt.loc["Cell"])
for i in range(1, dt.shape[0]):
    dict[dt.index[i]] = np.array(dt.loc[dt.index[i]], dtype=float)

savepath = 'case study/GSE72056/'


def sigmoid(x):
    return 1 / (1 + np.exp(-(x - 6)))


# 0=m,1=T,2=B,3=Macro,4=Endo,5=CAF,6=NK
malignant_index = np.where(dict['malignant'] == 2)[0]
T_index = np.where(dict['non-malignant cell type'] == 1)[0]
B_index = np.where(dict['non-malignant cell type'] == 2)[0]
Macro_index = np.where(dict['non-malignant cell type'] == 3)[0]
Endo_index = np.where(dict['non-malignant cell type'] == 4)[0]
CAF_index = np.where(dict['non-malignant cell type'] == 5)[0]
NK_index = np.where(dict['non-malignant cell type'] == 6)[0]


for i in range(7):
    for j in range(7):
        exec('mult_score{}{} = 0'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('thrd_score{}{} = 0'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('mult_list{}{} = []'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('mult_list_s{}{} = []'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('thrd_list{}{} = []'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('thrd_list_s{}{} = []'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('spec_score{}{} = 0'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('spec_list{}{} = []'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('spec_list_s{}{} = []'.format(i, j))



start_time = time.time()
process = psutil.Process()
memory_usage_start = process.memory_info().rss / (1024 * 1024)  # 转换为MB


for i in LRI_gene:
    if i[0] in dict and i[1] in dict:

        malignant_l = 1 / malignant_index.shape[0] * sum(dict[i[0]][malignant_index])
        malignant_r = 1 / malignant_index.shape[0] * sum(dict[i[1]][malignant_index])
        T_l = 1 / T_index.shape[0] * sum(dict[i[0]][T_index])
        T_r = 1 / T_index.shape[0] * sum(dict[i[1]][T_index])
        B_l = 1 / B_index.shape[0] * sum(dict[i[0]][B_index])
        B_r = 1 / B_index.shape[0] * sum(dict[i[1]][B_index])
        Macro_l = 1 / Macro_index.shape[0] * sum(dict[i[0]][Macro_index])
        Macro_r = 1 / Macro_index.shape[0] * sum(dict[i[1]][Macro_index])
        Endo_l = 1 / Endo_index.shape[0] * sum(dict[i[0]][Endo_index])
        Endo_r = 1 / Endo_index.shape[0] * sum(dict[i[1]][Endo_index])
        CAF_l = 1 / CAF_index.shape[0] * sum(dict[i[0]][CAF_index])
        CAF_r = 1 / CAF_index.shape[0] * sum(dict[i[1]][CAF_index])
        NK_l = 1 / NK_index.shape[0] * sum(dict[i[0]][NK_index])
        NK_r = 1 / NK_index.shape[0] * sum(dict[i[1]][NK_index])
        l_list = [malignant_l, T_l, B_l, Macro_l, Endo_l, CAF_l, NK_l]
        r_list = [malignant_r, T_r, B_r, Macro_r, Endo_r, CAF_r, NK_r]

        a = b = 0
        for item in product(l_list, r_list):  # product(A, B) 和 ((x,y) for x in A for y in B)一样
            # print("sigmoid:%f"%sigmoid(item[0]*item[1]))

            exec('mult_score{}{} += {}'.format(a, b, (item[0] * item[1])))
            exec('mult_list{}{}.append("{}" + "-" + "{}")'.format(a, b, i[0], i[1]))
            exec('mult_list_s{}{}.append({})'.format(a, b, (item[0] * item[1])))
            b += 1
            if b == 7:
                b = 0
                a += 1

        mean_l_malignant = np.mean(dict[i[0]][malignant_index])
        mean_l_T = np.mean(dict[i[0]][T_index])
        mean_l_B = np.mean(dict[i[0]][B_index])
        mean_l_Macro = np.mean(dict[i[0]][Macro_index])
        mean_l_Endo = np.mean(dict[i[0]][Endo_index])
        mean_l_CAF = np.mean(dict[i[0]][CAF_index])
        mean_l_NK = np.mean(dict[i[0]][NK_index])
        mean_l = np.mean((mean_l_malignant, mean_l_T, mean_l_B, mean_l_Macro, mean_l_Endo, mean_l_CAF, mean_l_NK))
        std_l = np.std(dict[i[0]][np.concatenate(
            (malignant_index, T_index, B_index, Macro_index, Endo_index, CAF_index, NK_index))])

        sum_l = np.sum((mean_l_malignant, mean_l_T, mean_l_B, mean_l_Macro, mean_l_Endo, mean_l_CAF, mean_l_NK))


        mean_r_malignant = np.mean(dict[i[1]][malignant_index])
        mean_r_T = np.mean(dict[i[1]][T_index])
        mean_r_B = np.mean(dict[i[1]][B_index])
        mean_r_Macro = np.mean(dict[i[1]][Macro_index])
        mean_r_Endo = np.mean(dict[i[1]][Endo_index])
        mean_r_CAF = np.mean(dict[i[1]][CAF_index])
        mean_r_NK = np.mean(dict[i[1]][NK_index])
        mean_r = np.mean((mean_r_malignant, mean_r_T, mean_r_B, mean_r_Macro, mean_r_Endo, mean_r_CAF, mean_r_NK))
        std_r = np.std(dict[i[1]][np.concatenate(
            (malignant_index, T_index, B_index, Macro_index, Endo_index, CAF_index, NK_index))])
        sum_r = np.sum((mean_l_malignant, mean_l_T, mean_l_B, mean_l_Macro, mean_l_Endo, mean_l_CAF, mean_l_NK))

        malignant_l = int(mean_l_malignant > mean_l + std_l)
        malignant_r = int(mean_r_malignant > mean_r + std_r)
        T_l = int(mean_l_T > mean_l + std_l)
        T_r = int(mean_r_T > mean_r + std_r)
        B_l = int(mean_l_B > mean_l + std_l)
        B_r = int(mean_r_B > mean_r + std_r)
        Macro_l = int(mean_l_Macro > mean_l + std_l)
        Macro_r = int(mean_r_Macro > mean_r + std_r)
        Endo_l = int(mean_l_Endo > mean_l + std_l)
        Endo_r = int(mean_r_Endo > mean_r + std_r)
        CAF_l = int(mean_l_CAF > mean_l + std_l)
        CAF_r = int(mean_r_CAF > mean_r + std_r)
        NK_l = int(mean_l_NK > mean_l + std_l)
        NK_r = int(mean_r_NK > mean_r + std_r)
        l_list = [malignant_l, T_l, B_l, Macro_l, Endo_l, CAF_l, NK_l]
        r_list = [malignant_r, T_r, B_r, Macro_r, Endo_r, CAF_r, NK_r]
        a = b = 0
        for item in product(l_list, r_list):
            exec('thrd_score{}{} += {}'.format(a, b, int(item[0] & item[1])))
            exec('thrd_list{}{}.append("{}" + "-" + "{}")'.format(a, b, i[0], i[1]))
            exec('thrd_list_s{}{}.append({})'.format(a, b, int(item[0] & item[1])))
            b += 1
            if b == 7:
                b = 0
                a += 1

        # 特异性
        sp_l_malignant = mean_l_malignant / sum_l
        sp_l_T = mean_l_T / sum_l
        sp_l_B = mean_l_B / sum_l
        sp_l_Macro = mean_l_Macro / sum_l
        sp_l_Endo = mean_l_Endo / sum_l
        sp_l_CAF = mean_l_CAF / sum_l
        sp_l_NK = mean_l_NK / sum_l
        sp_l_list = [sp_l_malignant, sp_l_T, sp_l_B, sp_l_Macro, sp_l_Endo, sp_l_CAF, sp_l_NK]

        sp_r_malignant = mean_r_malignant / sum_r
        sp_r_T = mean_r_T / sum_r
        sp_r_B = mean_r_B / sum_r
        sp_r_Macro = mean_r_Macro / sum_r
        sp_r_Endo = mean_r_Macro / sum_r
        sp_r_CAF = mean_r_CAF / sum_r
        sp_r_NK= mean_r_NK / sum_r
        sp_r_list = [sp_r_malignant, sp_r_T, sp_r_B, sp_r_Macro, sp_r_Endo, sp_r_CAF, sp_r_NK]

        a = b = 0
        for item in product(sp_l_list, sp_r_list):  # product(A, B) 和 ((x,y) for x in A for y in B)一样
            # print("sigmoid:%f"%sigmoid(item[0]*item[1]))

            exec('spec_score{}{} += {}'.format(a, b, (item[0] * item[1])))
            exec('spec_list{}{}.append("{}" + "-" + "{}")'.format(a, b, i[0], i[1]))
            exec('spec_list_s{}{}.append({})'.format(a, b, (item[0] * item[1])))
            b += 1
            if b == 7:
                b = 0
                a += 1



for i in range(7):
    for j in range(7):
        with open(savepath + "mult_score.txt", "a") as f:
            exec('f.write("mult_score{}{} = %f"%(mult_score{}{}))'.format(i, j, i, j))
            f.write('\n')
for i in range(7):
    for j in range(7):
        with open(savepath + "thrd_score.txt", "a") as f:
            exec('f.write("thrd_score{}{} = %f"%(thrd_score{}{}))'.format(i, j, i, j))
            f.write('\n')

for i in range(7):
    for j in range(7):
        exec('x = pd.DataFrame(mult_list{}{})'.format(i, j))
        exec('y = pd.DataFrame(mult_list_s{}{},columns=list("3"))'.format(i, j))
        x = x.join(y)
        exec('x.to_csv(savepath + "mult_list{}{}.csv", header=None, index=None)'.format(i, j))

for i in range(7):
    for j in range(7):
        exec('x = pd.DataFrame(thrd_list{}{})'.format(i, j))
        exec('y = pd.DataFrame(thrd_list_s{}{},columns=list("3"))'.format(i, j))
        x = x.join(y)
        exec('x.to_csv(savepath + "thrd_list{}{}.csv", header=None, index=None)'.format(i, j))
for i in range(7):
    for j in range(7):
        with open(savepath + "spec_score.txt", "a") as f:
            exec('f.write("spec_score{}{} = %f"%(spec_score{}{}))'.format(i, j, i, j))
            f.write('\n')

for i in range(7):
    for j in range(7):
        exec('x = pd.DataFrame(spec_list{}{})'.format(i, j))
        exec('y = pd.DataFrame(spec_list_s{}{},columns=list("3"))'.format(i, j))
        x = x.join(y)
        exec('x.to_csv(savepath + "spec_list{}{}.csv", header=None, index=None)'.format(i, j))

mm = MinMaxScaler()
mult = []
thrd = []
spec = []
for i in range(7):
    for j in range(7):
        exec('mult.append(mult_score{}{})'.format(i, j))
        exec('thrd.append(thrd_score{}{})'.format(i, j))
        exec('spec.append(spec_score{}{})'.format(i, j))
mult = np.array(mult).reshape(7, 7)
thrd = np.array(thrd).reshape(7, 7)
spec = np.array(spec).reshape(7, 7)

multy = pd.DataFrame(mult)
thrdy = pd.DataFrame(thrd)
specy = pd.DataFrame(spec)

multy.to_csv(savepath + "CCCmulty.csv", header=None, index=None)
thrdy.to_csv(savepath + "CCCthrdy.csv", header=None, index=None)
specy.to_csv(savepath + "CCCspecy.csv", header=None, index=None)

mult = mm.fit_transform(mult)
thrd = mm.fit_transform(thrd)
spec = mm.fit_transform(spec)

process = psutil.Process()
da = []
for i in range(7):
    for j in range(7):
        a = mult[i][j]
        b = thrd[i][j]
        c = spec[i][j]

        result = (a+b+c) / 3
        da.append(result)
da = np.array(da).reshape(7, 7)
da = pd.DataFrame(da)
da.to_csv(savepath + "CCCscore_result.csv", header=None, index=None)



