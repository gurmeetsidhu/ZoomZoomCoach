import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import datetime
import argparse
import pickle as pkl
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd
import os
import cv2
import sys
import glob
import time

# As per https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

from threading import Thread

if sys.version_info >= (3, 0):
	from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
 
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
 
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				_,_ = self.stream.read()
				grabbed, frame = self.stream.read()
 
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					print("Not grabbed")
					frame = None
					if self.stream.get(CV_CAP_PROP_POS_FRAMES)==self.stream.get(7)-1:
						self.stop()
					return
 
				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		print("Stopped")
		# indicate that the thread should be stopped
		self.stopped = True

#Take split track and convert it into an evenly spaced inner and outer limit
def Spline_Track(track, segments, insidex, insidey, outsidex, outsidey):

	insidex, insidey, outsidex, outsidey = track[0][0], track[0][1], track[1][0], track[1][1]

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

def Read_Main_CSV(saveFile, replayFile="ignore"):
	data = pd.read_csv(saveFile)
	laps = data["mCurrentLap"].unique()
	for lap in laps:
		lapData = data.loc[data["mCurrentLap"]==lap]
		car = lapData["mCarName"].unique()[0]
		trackLocation = lapData["mTrackLocation"].unique()[0]
		date = datetime.datetime.now().strftime("%m-%d-%Y")
		lapTime = lapData["mCurrentTime"].max()
		saveDir = "./backups/" + trackLocation + "/" + car + "/" + date + "/"
		saveName = "lap" + str(lap) + "-" + str(lapTime) + ".csv"
		if not os.path.exists(saveDir):
			# Directory doesn't exist creating directory
			os.makedirs(saveDir)
		lapData.to_csv((saveDir+saveName), index=False)

	#Need to write function to split up screen recordings into seperate files for each lap

def Visualize_CSV(visFile):
	data = pd.read_csv(visFile)
	laps = data["mCurrentLap"].unique()
	plt.figure(0)
	for lap in laps:
		lapData = data.loc[data["mCurrentLap"]==lap]
		x,y = lapData["mWorldPosition[0]"], lapData["mWorldPosition[2]"]
		plt.plot(x, y)
	plt.show()

def Test_Visualize_CSV(loadFile, visFile):
	print("Loading video file into threaded memory...")
	cap = FileVideoStream(visFile,1024)
	LapData = pd.read_csv(loadFile)
	vidLength = cap.stream.get(7)-1

	print("Please close the frame that is opened once you have the lap time in the top left in seconds. (i.e. 2:07:080 => 127.080)")	
	cap.stream.set(1,0)
	ret, frame = cap.stream.read()
	print(ret)
	cv2.imshow("Please write current time in seconds with decimal in top left",frame)
	cv2.waitKey(0)
	minTime = float(raw_input("Time in seconds with decimals:"))
	LapData = LapData.loc[LapData["mCurrentTime"]>=minTime]

	print("Please close the frame that is opened once you have the lap time in the top left in seconds. (i.e. 2:07:080 => 127.080)")
	cap.stream.set(1, vidLength-1)
	ret, frame = cap.stream.read()
	print(ret)
	cv2.imshow("Please write current time in seconds with decimal in top left",frame)
	cv2.waitKey(0)
	maxTime = float(raw_input("Time in seconds with decimals:"))
	LapData = LapData.loc[LapData["mCurrentTime"]<=maxTime]

	cap.stream.set(1,0)
	cap.start()
	time.sleep(1)

	trackLocation = LapData["mTrackLocation"].unique()[0]

	print("Loading" + trackLocation + "splined track. From address: " + "./backups/" + trackLocation + "/Track-Splined.pkl")
	with open("./backups/" + trackLocation + "/Track-Splined.pkl", "rb") as rf:
		track = pkl.load(rf)

	speedArr = LapData["mSpeed"]*3.6
	timeArr = LapData["mCurrentTime"]
	#Find how long 10 secs takes using refresh rate.
	
	refreshwindow = np.argmax(timeArr>10)

	raceSectors = [0]
	currentSector = 1
	for i in range(0,(len(speedArr)-1)-refreshwindow):
		maxIndex = speedArr[i:(refreshwindow-1+i)].argmax()
		if maxIndex <= (i+0.6*(refreshwindow-1)) and maxIndex >= (i+0.4*(refreshwindow-1)):
			if maxIndex not in raceSectors:
				raceSectors.append(maxIndex)
	# for i in range(0, len(raceSectors)-1):

	print("Preparing matplot figure and data...")
	fig = plt.figure(figsize=(16,9), dpi=120)
	gs = gridspec.GridSpec(9,16)
	gs.update(wspace=0, hspace=0)
	ax1 = plt.subplot(gs[0:4,0:4])
	ax2 = plt.subplot(gs[0:2,4:10])
	ax3 = plt.subplot(gs[0:2,10:16])
	ax4 = plt.subplot(gs[2:4,4:10])
	ax5 = plt.subplot(gs[2:4,10:16])
	ax6 = plt.subplot(gs[4:9,0:10])
	ax7 = plt.subplot(gs[4:6,10:16])
	ax8 = plt.subplot(gs[6:9,10:16])
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	charts = [ax1, ax6]
	graphs = [ax2, ax3, ax4, ax5, ax7, ax8]
	for chart in charts:
		chart.axes.get_xaxis().set_visible(False)
		chart.axes.get_yaxis().set_visible(False)

	for graph in graphs:
		graph.tick_params(axis="x", direction="in", pad=-10, labelsize=8)
		graph.tick_params(axis="y", direction="in", pad=-22, gridOn=True, labelsize=8)
		graph.set_xlim([timeArr.min(),timeArr[raceSectors[1]]])

	ax1.plot(track[0][0], track[0][1], 'b', alpha=0.2)
	ax1.plot(track[1][0], track[1][1], 'b', alpha=0.2)
	track = None
	xmin, xmax = LapData["mWorldPosition[0]"][raceSectors[0]:raceSectors[1]].min(),LapData["mWorldPosition[0]"][raceSectors[0]:raceSectors[1]].max()
	ymin, ymax = LapData["mWorldPosition[2]"][raceSectors[0]:raceSectors[1]].min(),LapData["mWorldPosition[2]"][raceSectors[0]:raceSectors[1]].max()
	mapSize = max(xmax-xmin,ymax-ymin)+5
	xavg = (xmax+xmin)/2
	yavg = (ymax+ymin)/2
	ax1.set_xlim([xavg-mapSize, xavg+mapSize])
	ax1.set_ylim([yavg-mapSize, yavg+mapSize])
	points = np.array([LapData["mWorldPosition[0]"], LapData["mWorldPosition[2]"]]).T.reshape(-1,1,2)
	segments = np.concatenate([points[:-1],points[1:]], axis=1)
	
	norm = plt.Normalize(speedArr.min(), speedArr.max())
	lc = LineCollection(segments, cmap='viridis', norm=norm)
	lc.set_array(speedArr[1:])
	lc.set_linewidth(2)
	line = ax1.add_collection(lc)

	line1, = ax2.plot(timeArr, speedArr, "r")
	line2, = ax3.plot(timeArr, -LapData["mSteering"], "r")
	line3, = ax4.plot(timeArr, LapData["mLocalAcceleration[2]"]/9.8, 'b', alpha=0.2)
	line4, = ax4.plot(timeArr, LapData["mBrake"], 'r')
	line5, = ax4.plot(timeArr, LapData["mThrottle"], 'g')
	line6, = ax5.plot(timeArr, LapData["mLocalAcceleration[0]"]/9.8)
	line7, = ax7.plot(timeArr, LapData["mOrientation[0]"], "r", alpha=0.6)
	line8, = ax7.plot(timeArr, LapData["mOrientation[1]"], "g", alpha=0.6)
	line9, = ax7.plot(timeArr, LapData["mOrientation[2]"], "b", alpha=0.6)
	line10, = ax8.plot(timeArr, LapData["mAngularVelocity[0]"], "r", alpha=0.6)
	line11, = ax8.plot(timeArr, LapData["mAngularVelocity[1]"], "g", alpha=0.6)
	line12, = ax8.plot(timeArr, LapData["mAngularVelocity[2]"], "b", alpha=0.6)

	print("Starting while loop to display video...")
	x=0
	telemetryLength = len(timeArr)
	sectorStart = raceSectors[0]
	sectorEnd = raceSectors[1]

	video = cv2.VideoWriter('analysis.mp4',cv2.VideoWriter_fourcc('H','2','6','4'),30,(1920,1080))

	while cap.more():
		completionPerc = int(x/(vidLength/2)*telemetryLength)
		if completionPerc>sectorEnd:
			currentSector+=1
			sectorStart = raceSectors[currentSector-1]
			try:
				sectorEnd = raceSectors[currentSector]
			except:
				sectorEnd = telemetryLength-1
			for graph in graphs:
				graph.set_xlim([timeArr[sectorStart],timeArr[sectorEnd]])
			xmin, xmax = LapData["mWorldPosition[0]"][sectorStart:sectorEnd].min(),LapData["mWorldPosition[0]"][sectorStart:sectorEnd].max()
			ymin, ymax = LapData["mWorldPosition[2]"][sectorStart:sectorEnd].min(),LapData["mWorldPosition[2]"][sectorStart:sectorEnd].max()
			mapSize = max(xmax-xmin,ymax-ymin)+5
			xavg = (xmax+xmin)/2
			yavg = (ymax+ymin)/2
			ax1.set_xlim([xavg-mapSize, xavg+mapSize])
			ax1.set_ylim([yavg-mapSize, yavg+mapSize])

		line1.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(speedArr[sectorStart:completionPerc]))
		line2.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(-LapData["mSteering"][sectorStart:completionPerc]))
		line3.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mLocalAcceleration[2]"][sectorStart:completionPerc]/9.8))
		line4.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mBrake"][sectorStart:completionPerc]))
		line5.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mThrottle"][sectorStart:completionPerc]))
		line6.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mLocalAcceleration[0]"][sectorStart:completionPerc]/9.8))
		line7.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mOrientation[0]"][sectorStart:completionPerc]))
		line8.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mOrientation[1]"][sectorStart:completionPerc]))
		line9.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mOrientation[2]"][sectorStart:completionPerc]))
		line10.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mAngularVelocity[0]"][sectorStart:completionPerc]))
		line11.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mAngularVelocity[1]"][sectorStart:completionPerc]))
		line12.set_data(np.array(timeArr[sectorStart:completionPerc]),np.array(LapData["mAngularVelocity[2]"][sectorStart:completionPerc]))

		# start = time.time()
		# redraw the canvas
		fig.canvas.draw()

		# convert canvas to image
		img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

		# img is rgb, convert to opencv's default bgr
		img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

		# display image with opencv or any operation you like

		# here = time.time()
		# print("It took " + str(here-start) + " seconds to prep figure here")
		# display camera feed
		frame = cap.read()
		if frame != None:
			img[510:1050,120:1080] = frame
			# cv2.imshow("cam",img)
			video.write(img)

		# here = time.time()
		# print("It took " + str(here-start) + " seconds to display video here")
		k = cv2.waitKey(15) & 0xFF
		if k == 27:
			break

		x+=1
		print("Wrote frame " + str(x) + " of " + str(vidLength/2) + ". Currently on sector " + str(currentSector) + " of " + str(len(raceSectors)))
	cv2.destroyAllWindows()
	video.release()

def main():

	try:
		list_of_files = glob.glob('./ScreenRecordings/*')
		latest_file = max(list_of_files, key=os.path.getctime)
	except ValueError:
		latest_file = "Doesn't Exist"
		print("WARNING! There is no screen recordings. If trying to -save and recordings are not in ./ScreenRecordings/ folder please restart script ctrl+c and try again once replays are loaded in folder")

	parser = argparse.ArgumentParser(description="Welcome to ZoomZoom Coach. Use this function to process and analyze your laps after recording.")
	parser.add_argument("-save", help="Save recorded lap into backup folders", dest="saveFile", type=str)
	parser.add_argument("-load", help="Load recorded lap/folder to analyze", dest="loadFile", type=str)
	parser.add_argument("-track", help="Save recorded laps as inner/outer track limits respecively", dest="trackFile", type=str)
	parser.add_argument("-replay", help="Location of replay file", dest="replayFile", type=str, default=latest_file)
	parser.add_argument("-vis", help="Use this to visualize your inner/outer tracks to make sure they are seperate laps and unnecessary points are deleted.", dest="visFile", type=str)
	args=parser.parse_args()

	if (args.visFile and args.loadFile):
		Test_Visualize_CSV(args.loadFile, args.visFile)
		# Visualize_CSV(args.visFile)

	if (args.saveFile):
		if (os.path.isfile(args.replayFile)) == False:
			print("WARNING:You are trying to perform -save function with no recordings. Do you want to exit the script and save recordings before continuing (Y/N)")
			x = raw_input()
			if (str(x)[0].lower() == "n"):
				Read_Main_CSV(args.saveFile)
			else:
				sys.exit()
		else:
			Read_Main_CSV(args.saveFile, args.replayFile)

	# if (args.loadFile and args.replayFile):
	# 	Test_Visualize_CSV(args.loadFile)

if __name__ == '__main__':
	main()