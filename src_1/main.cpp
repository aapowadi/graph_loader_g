
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
	while (1)
	{
		cout << "update\n" << endl;
		g_pCap->getRGBFrame(frame);
		g_pCap->getDepthFrame(depth);
		cv::imshow("frame", frame);
		cv::waitKey(1);
		//	cv::imshow("depth", depth);
		//	cv::waitKey(1);
		//	vector<cv::String> fn;
		//	glob("C:/Demo/AFRL-SpatialReg/RVDemo/vs2015x64/plane_images/*.png", fn, false);

		//	vector<Mat> images;
		//	size_t count = fn.size(); //number of png files in images folder
		//	for (int i = 0; i<count; i++)
		//	images.push_back(imread(fn[i]));

		//	for (int j = 0; j < count; j++)
		//	{
		bool new_pose = false;
		while (new_pose == false)
		{
			new_pose = new_frame.getpose(frame, depth, position);
		}
		//ut << "\nTranslation:"<<;
		for (int i = 2; i >= 0; i--)
		{
			cout << position[i] << ",";
		}
		/*	cout << "\n"<<"rotational pose: ";

			for (int i = 6; i >= 3; i--)
			{
				cout << position[i] << ",";
			}
			cout << "\n";*/
			//	}
	}
		return 1;
}