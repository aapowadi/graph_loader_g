#pragma once

#include "tf_utils.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <tuple>
#include <vector>

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
	struct Pt {
		double x, y; //coordinates
		int cluster; //no default cluster
		double minDist; //default distance = 2

		Pt() :
			x(0.0),
			y(0.0),
			cluster(-1),
			minDist(2) {}

		Pt(double x, double y) :
			x(x),
			y(y),
			cluster(-1),
			minDist(2) {}

		double distance(Pt p) {
			return (p.x - x)*(p.x - x) + (p.y - y)*(p.y - y);
		}
	};
	vector<int> nPoints;
	void deleteTensor(TF_Tensor* pTensor) {
		if (pTensor)
			tf_utils::DeleteTensor(pTensor);
	}
	void gen_bnd_box(const cv::Mat& img, cv::Mat& bnd_img, int scale);
	void normalizergb(const cv::Mat& img, cv::Mat& edges);
	void loadImgData(const cv::Mat edges, vector<float_t>& im_data, vector<int64_t>& im_dims);
	void load1chData(const cv::Mat edges, vector<float_t>& im_data, vector<int64_t>& im_dims);
	void link1_clust(vector<Pt>* points, vector<Pt> centroids);
	TF_Operation* getOp(TF_Graph* graph, const std::string op_name);

	string _workingDirectory = "";

		
	TF_Session* _pSession = nullptr;
	bool _sessionInitialized = false;

	TF_Status* _pStatus = nullptr;
	int count;
	TF_Tensor* _pInputColorTensor = nullptr;
	TF_Tensor* _pInputSegmentedTensor = nullptr;
	TF_Tensor* _pInputDepthTensor = nullptr;
	TF_Tensor* _pDropoutTensor = nullptr;

	TF_Graph* _pGraph = nullptr;
};