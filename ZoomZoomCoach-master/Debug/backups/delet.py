import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pickle as pkl

# plt.figure()
track = pd.read_csv("./mapLog.csv")
laps = track["mCurrentLap"].unique()
for lap in laps:
	track.loc[track["mCurrentLap"]==lap]
	filename = "./mapLog" + str(lap) + ".csv"
	track.to_csv(filename, index=False)

# with open("stupidArr.pkl", "rb") as file:
# 	stupidArr = pkl.load(file)

# x = stupidArr[0]
# z = stupidArr[1]
# plt.plot(x,z)


# track2 = pd.read_csv("C:/Users/Baltej/Documents/SMS_MemMap_Sample_V8/Debug/backups/Laguna/BRZ/12-8-2018/mapLog2.csv")
# print(track2)

# x1 = (track2['mWorldPosition[0]'])
# y1 = (track2['mWorldPosition[1]'])
# z1 = (track2['mWorldPosition[2]'])
# plt.plot(x1,z1)
# plt.show()