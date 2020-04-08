_Companyname_ = "Foundingminds"
_Author_ = "Amrit Sreekumar"

import facematch
import cv2

def main():

	#the path of the images that need to be compared comes here
	img1  = "harry_old.png"
	img2 = "harry_young.jpeg"


	#reading images
	img1 = cv2.imread(img1)
	img2 = cv2.imread(img2)

	#finding the match
	facematch.match_face(img1,img2)

main()