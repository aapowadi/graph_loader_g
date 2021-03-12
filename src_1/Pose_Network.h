#pragma once

#include "tf_utils.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

class Pose_Network
{
public:
	Pose_Network(string workingDirectory = ".");
	virtual ~Pose_Network();

	/**
	 * Gets the position of the probe
	 * @param in_img The color image
	 * @param depth The depth image
	 * @param position The position of the probe
	 */
	bool getpose(const Mat& in_img, const Mat& depth, cv::Vec3f& position);

	/**
	 * Returns if the session has been initialized
	 */
	bool isInitialized() { return _sessionInitialized; }


private:

	void deleteTensor(TF_Tensor* pTensor) {
		if (pTensor)
			tf_utils::DeleteTensor(pTensor);
	}

	void loadImgData(const cv::Mat edges, vector<float_t>& im_data, vector<int64_t>& im_dims);
	void load1chData(const cv::Mat edges, vector<float_t>& im_data, vector<int64_t>& im_dims);
	TF_Operation* getOp(TF_Graph* graph, const std::string op_name);

	string _workingDirectory = "";

		
	TF_Session* _pSession = nullptr;
	bool _sessionInitialized = false;

	TF_Status* _pStatus = nullptr;

	TF_Tensor* _pInputColorTensor = nullptr;
	TF_Tensor* _pInputSegmentedTensor = nullptr;
	TF_Tensor* _pInputDepthTensor = nullptr;
	TF_Tensor* _pDropoutTensor = nullptr;

	TF_Graph* _pGraph = nullptr;
};