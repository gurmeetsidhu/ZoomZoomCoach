import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import pickle as pkl
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd
import os

#Take split track and convert it into an evenly spaced inner and outer limit
def Spline_Track(segments, insidex, insidey, outsidex, outsidey):

	insidex, insidey, outsidex, outsidey = laguna[0][0], laguna[0][1], laguna[1][0], laguna[1][1]

	idsx = np.arange(len(insidex))//25
	resultx = np.bincount(idsx,insidex)/np.bincount(idsx)
	idsy = np.arange(len(insidey))//25
	resulty = np.bincount(idsy,insidey)/np.bincount(idsy)

	ptsInner = np.array(zip(resultx, resulty))

	tck, u = splprep(ptsInner.T, u=None, s=0.0, per=1) 
	u_inner = np.linspace(u.min(), u.max(), segments)
	x_inner, y_inner = splev(u_inner, tck, der=0)
	x_inner = np.append(x_inner[4:], x_inner[:4])
	y_inner = np.append(y_inner[4:], y_inner[:4])

	odsx = np.arange(len(outsidex))//25
	resultx = np.bincount(odsx,outsidex)/np.bincount(odsx)
	odsy = np.arange(len(outsidey))//25
	resulty = np.bincount(odsy,outsidey)/np.bincount(odsy)

	ptsOuter = np.array(zip(resultx, resulty))

	tck, u = splprep(ptsOuter.T, u=None, s=0.0, per=1) 
	u_outer = np.linspace(u.min(), u.max(), segments)
	x_outer, y_outer = splev(u_outer, tck, der=0)

	return x_inner, y_inner, x_outer, y_outer

#Save track limits to file 
def Output_Track_Limits(fname, x_inner, y_inner, x_outer, y_outer):
	with open(fname, "wb") as of:
		pkl.dump([[x_inner, y_inner],[x_outer, y_outer]], of)


def main():

	with open("./backups/Laguna/Laguna-Splined.pkl", "rb") as rf:
		laguna = pkl.load(rf)

	for file in os.listdir("./backups/Laguna/BRZ/12-8-2018/"):
		LapData = pd.read_csv("./backups/Laguna/BRZ/12-8-2018/" + file)
		speedArr = LapData["mSpeed"]*3.6
		timeArr = LapData["mCurrentTime"]
		#Find how long 5 secs takes using refresh rate.
		refreshwindow = np.argmax(timeArr>10)

		raceSectors = [0]
		for i in range(0,(len(speedArr)-1)-refreshwindow):
			maxIndex = speedArr[i:(refreshwindow-1+i)].argmax()
			if maxIndex <= (i+0.6*(refreshwindow-1)) and maxIndex >= (i+0.4*(refreshwindow-1)):
				if maxIndex not in raceSectors:
					raceSectors.append(maxIndex)

		for i in range(0, len(raceSectors)-1):
			SecData = LapData[raceSectors[i]:raceSectors[i+1]]
			speedArr = SecData["mSpeed"]*3.6
			timeArr = SecData["mCurrentTime"]

			plt.figure(0)
			gs = gridspec.GridSpec(9,16)
			gs.update(wspace=0, hspace=0, left=0.1, bottom=0.1)
			ax1 = plt.subplot(gs[0:4,0:4])
			ax2 = plt.subplot(gs[0:2,4:10])
			ax3 = plt.subplot(gs[0:2,10:16])
			ax4 = plt.subplot(gs[2:4,4:10])
			ax5 = plt.subplot(gs[2:4,10:16])
			ax6 = plt.subplot(gs[4:9,0:10])
			ax7 = plt.subplot(gs[4:6,10:16])
			ax8 = plt.subplot(gs[6:9,10:16])
			charts = [ax1, ax6]
			graphs = [ax2, ax3, ax4, ax5, ax7, ax8]
			for chart in charts:
				chart.axes.get_xaxis().set_visible(False)
				chart.axes.get_yaxis().set_visible(False)

			for graph in graphs:
				graph.tick_params(axis="x", direction="in", pad=-10, labelsize=8)
				graph.tick_params(axis="y", direction="in", pad=-22, gridOn=True, labelsize=8)

			ax1.plot(laguna[0][0], laguna[0][1], 'b', alpha=0.2)
			ax1.plot(laguna[1][0], laguna[1][1], 'b', alpha=0.2)
			points = np.array([SecData["mWorldPosition[0]"], SecData["mWorldPosition[2]"]]).T.reshape(-1,1,2)
			segments = np.concatenate([points[:-1],points[1:]], axis=1)
			
			norm = plt.Normalize(speedArr.min(), speedArr.max())
			lc = LineCollection(segments, cmap='viridis', norm=norm)
			lc.set_array(speedArr[1:])
			lc.set_linewidth(2)
			line = ax1.add_collection(lc)

			ax2.plot(timeArr, speedArr, "r")
			ax3.plot(timeArr, -SecData["mSteering"], "r")
			ax4.plot(timeArr, SecData["mLocalAcceleration[2]"]/9.8, 'b', alpha=0.2)
			ax4.plot(timeArr, SecData["mBrake"], 'r')
			ax4.plot(timeArr, SecData["mThrottle"], 'g')
			ax5.plot(timeArr, SecData["mLocalAcceleration[0]"]/9.8)
			#ax6.plot(video?????)
			ax7.plot(timeArr, SecData["mOrientation[0]"], "r", alpha=0.6)
			ax7.plot(timeArr, SecData["mOrientation[1]"], "g", alpha=0.6)
			ax7.plot(timeArr, SecData["mOrientation[2]"], "b", alpha=0.6)
			ax8.plot(timeArr, SecData["mAngularVelocity[0]"], "r", alpha=0.6)
			ax8.plot(timeArr, SecData["mAngularVelocity[1]"], "g", alpha=0.6)
			ax8.plot(timeArr, SecData["mAngularVelocity[2]"], "b", alpha=0.6)
			#ax1.plot(SecData["mWorldPosition[0]"], SecData["mWorldPosition[2]"])#, c=SecData["mSpeed"], cmap=plt.get_cmap('viridis'))
			plt.show()

if __name__ == '__main__':
	main()