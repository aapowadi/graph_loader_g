
// STL
#include <memory>

// local
#include "RV10_Configs.h"
#include "StructureCoreCaptureDevice.h"
#include "Pose_Network.h"

Pose_Network new_frame;
cv::Vec3f position;
std::shared_ptr<afrl::StructureCoreCaptureDevice> g_pCap;
int main()
{
	if (!g_pCap)
		g_pCap = std::make_shared<afrl::StructureCoreCaptureDevice>();
		cv::Mat frame, depth;

	cv::VideoCapture cap("../bin/TRI_plane.mp4");
	if (!cap.isOpened())
	{
		cout << "Error streaming the video file" << endl;
		return -1;
	}
	while (1)
	{ 
		cout << "update\n" << endl;
		//g_pCap->getRGBFrame(frame);
		//g_pCap->getDepthFrame(depth);
		//cv::imshow("frame", frame);
		//cv::waitKey(1);
		
		/*cap >> frame;

		if (frame.empty())
			break;*/
		//cv::imshow("frame", frame);

		vector<cv::String> fn;
		glob("../bin/test/*.png", fn, false);

		vector<Mat> images;
		size_t count = fn.size(); //number of png files in images folder
		Mat temp_in;
		Mat temp_d;
	  
		for (int j = 0; j < count; j++)
		{

			frame = imread(fn[j]);
			/*imshow("temp-in", temp_in);
			waitKey(1);*/
			bool new_pose = false;
			
			while (new_pose == false)
			{
			
				new_pose = new_frame.getpose(frame, frame, position);
			}
			//ut << "\nTranslation:"<<;
			for (int i = 2; i >= 0; i--)
			{
				cout << position[i] << ",";
			}
			cout << "\n";
			/*	cout << "\n"<<"rotational pose: ";

				for (int i = 6; i >= 3; i--)
				{
					cout << position[i] << ",";
				}
				cout << "\n";*/
				//	}
		}
	}
		return 1;
}