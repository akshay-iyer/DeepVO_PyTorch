import os
import numpy as np
import glob

def getTransVector(files):
	#path = "/home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/pose_GT/00.txt"
	with open(files) as f:
			lines = [line.split('\n')[0] for line in f.readlines()]
			
			tVector = []
			for l in lines:
				temp = [float(val) for val in l.split(' ')] 
				t = [temp[3], temp[7], temp[11]]
				tVector.append(t)
	return tVector

			
# def R_to_angle(Rt):
# # Ground truth pose is present as [R | t] 
# # R: Rotation Matrix, t: translation vector
# # transform matrix to angles
# 	Rt = np.reshape(np.array(Rt), (3,4))
# 	t = Rt[:,-1]
# 	return t

def getEucDist(a,b):
	sum=0
	for i in range(len(a)):
		sum += (a[i] - b[i])**2
	sum = np.sqrt(sum)
	return sum	

if __name__ ==	 '__main__':
	# dir_path = "home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/test_paths/"
	# print ('hello')
	f = open("/home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/lengths.txt", "a+")
	for files in glob.iglob('/home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/test_paths/*.txt'):
		print(files)
		tVector = getTransVector(files)
		#print(tVector)
		length = 0.0
		for i in range(len(tVector) -1):
			length += getEucDist(tVector[i],tVector[i+1])
		print(length)
		f.write("{} : {}\n".format(files,str(length)))
	f.close()
	f = open("/home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/pose_GT/lengths.txt", "a+")
	for folder in os.listdir('/home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/images/'):
		folder = '/home/akshay/Documents/Books/DR/Code/DeepVO-pytorch-master/KITTI/images/'+folder
		for files in os.listdir(folder):
			if files == 'times.txt':
				g = open(folder+'/'+files,"+r")
				linelist  = g.readlines()
				f.write("{} : {}".format(folder,linelist[-1]))
				print(folder)
				print (linelist[-1])
				print(" ")

		