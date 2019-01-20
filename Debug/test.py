import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2
import matplotlib.pyplot as plt

from threading import Thread
import sys
import time

if sys.version_info >= (3, 0):
	from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue


class FileVideoStream:
	def __init__(self, path, queueSize=256):
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
				grabbed, frame = self.stream.read()
				grabbed, frame = self.stream.read()
 
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					print("Not grabbed")
					self.stop()
					return
 
				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		print(self.Q.qsize())
		return self.Q.qsize() > 0

	def stop(self):
		print("Stopped")
		# indicate that the thread should be stopped
		self.stopped = True

fig = plt.figure()
cap = FileVideoStream(".\\ScreenRecordings\\2018-12-24 13-01-13.mp4").start()
time.sleep(1)

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)


line1, = plt.plot(x1, y1, 'ko-')        # so that we can update data later
print(line1)

for i in range(0,1000):
	# update data
	y1 = np.cos(2 * np.pi * (x1+i*3.14/2) ) * np.exp(-x1)
	print(x1,y1)
	line1.set_ydata(np.cos(2 * np.pi * (x1+i*3.14/2) ) * np.exp(-x1) )

	# redraw the canvas
	fig.canvas.draw()

	# convert canvas to image
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	# img is rgb, convert to opencv's default bgr
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


	# display image with opencv or any operation you like
	cv2.imshow("plot",img)

	# display camera feed
	frame = cap.read()
	cv2.imshow("cam",frame)

	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break