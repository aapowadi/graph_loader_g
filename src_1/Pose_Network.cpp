/* Pose Estimation Class
	-by Anirudha Powadi
*/
#include"Pose_Network.h"


Pose_Network::Pose_Network(string workingDirectory) :
	_workingDirectory(workingDirectory)
{
	// Protobuf file of graph to loadsession
	std::string load_path(_workingDirectory + "/../../models/fix_Exp8_800.pb");

	_pStatus = TF_NewStatus();
    _pGraph = tf_utils::LoadGraph(load_path.c_str(), _pStatus);
    if (TF_GetCode(_pStatus) != TF_OK || _pGraph == nullptr) {
        std::cout << "Failed loading graph: " << TF_Message(_pStatus) << std::endl;
        return;
    }

	std::cout << "Success loading graph" << std::endl;
	
	// Create a Tensorflow session to execute graph in
    _pSession = tf_utils::CreateSession(_pGraph);
	if (_pSession == nullptr) {
		std::cout << "Couldn't make session" << std::endl;
		return;
	}
	count = 0;
	// Create Tensors
	//std::vector<int64_t> in_dims = { 1, 64, 64, 1 };
	//_pInputColorTensor = tf_utils::CreateEmptyTensor(TF_FLOAT, in_dims);
	//_pInputSegmentedTensor = tf_utils::CreateEmptyTensor(TF_FLOAT, in_dims);
	//_pInputDepthTensor = tf_utils::CreateEmptyTensor(TF_FLOAT, in_dims);
	//std::vector<int64_t> dropout_dims = { 1 };
	//_pDropoutTensor = tf_utils::CreateEmptyTensor(TF_FLOAT, dropout_dims);

	_sessionInitialized = true;
}

Pose_Network::~Pose_Network()
{
	// Delete TensorFlow artifacts
	if (_pSession) tf_utils::DeleteSession(_pSession);
	//if (_pInputColorTensor) tf_utils::DeleteTensor(_pInputColorTensor);
	//if (_pInputSegmentedTensor) tf_utils::DeleteTensor(_pInputSegmentedTensor);
	//if (_pInputDepthTensor) tf_utils::DeleteTensor(_pInputDepthTensor);
	//if (_pDropoutTensor) tf_utils::DeleteTensor(_pDropoutTensor);
	if (_pGraph) tf_utils::DeleteGraph(_pGraph);
	if (_pStatus) TF_DeleteStatus(_pStatus);
}

/**
 * Load an 8-bit RGB image from a path into a raw buffer in the correct shape to match the tensor memory layout
 */
void Pose_Network::loadImgData(const cv::Mat edges, vector<float_t>& im_data, vector<int64_t>& im_dims) 
{
	im_data.clear();
	im_dims.clear();

	cv::Mat loaded_img = edges;
	// make 8-bit three channel
	//edges.convertTo(loaded_img, CV_8UC3);

	// Dimensions [batch, height, width, channels]
	im_dims = { 1, loaded_img.rows, loaded_img.cols, 3 };
	for (int y = 0; y < loaded_img.rows; ++y) {
		for (int x = 0; x < loaded_img.cols; ++x) {
			auto pixel = loaded_img.at<cv::Vec3b>(y, x);
			for (int d = 0; d < 3; ++d) {
				im_data.push_back(pixel[d]);
			}
		}
	}
	loaded_img.release();
}

void Pose_Network::load1chData(const cv::Mat edges, vector<float_t>& im_data, vector<int64_t>& im_dims)
{
	im_data.clear();
	im_dims.clear();

	cv::Mat loaded_res1 = edges;
	// make 8-bit 1 channel
	//loaded_res1.convertTo(loaded_res1, CV_8UC1);
	//Mat loaded_res1 = edges.reshape(0, 64);

	// Dimensions [batch, height, width, channels]
	im_dims = { 1, loaded_res1.rows, loaded_res1.cols, 1 };
	for (int y = 0; y < loaded_res1.rows; ++y) {
		for (int x = 0; x < loaded_res1.cols; ++x) {
			auto pixel1 = loaded_res1.at<float>(y, x);
				im_data.push_back(pixel1);
		}
	}
	loaded_res1.release();
}
/**
 * Loads an op from the graph and checks that it loaded successfully
 */
TF_Operation* Pose_Network::getOp(TF_Graph* _pGraph, const std::string op_name) {
	auto op = TF_GraphOperationByName(_pGraph, op_name.c_str());
	if (op == nullptr) 
	{
       std::cerr << "Unable to load op \"" << op_name << "\"" << std::endl;
	}
	return op;
}

void Pose_Network::link1_clust(vector<Pt>* points, vector<Pt> centroids)
{
	//Assigning points to a cluster
	nPoints.clear();
	for (vector<Pt>::iterator c = begin(centroids);
		c != end(centroids); ++c) {
		//quick hack to get clusterID
		int clusterId = c - begin(centroids);

		for (vector<Pt>::iterator it = points->begin();
			it != points->end(); ++it) {
			Pt p = *it;
			double dist = c->distance(p);

			if (dist <= p.minDist)
			{
				p.cluster = clusterId;
			}

			*it = p;
		}
	}
	for (int j = 0; j < sizeof(centroids); j++) {
		nPoints.push_back(0);
	}

	for (vector<Pt>::iterator it = points->begin();
		it != points->end(); ++it) {
		int clusterId = it->cluster;
		nPoints[clusterId] += 1;
	}
}

void Pose_Network::gen_bnd_box(const cv::Mat& img, cv::Mat& bnd_img, int scale)
{
	vector<Pt> seed_pts, v_points;
	Pt locs;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int k_ones = 1;
			if (k_ones & img.at<uchar>(Point(i, j)) == 1)
			{
				locs.x = i; locs.y = j;
				v_points.push_back(locs);
			}
			int res = 0;
			if ((i + 1) % scale == 0 && (j + 1) % scale == 0)
			{
				for (int k = i; k < i + scale; k++)
				{
					if ((k + scale) > img.rows)
					{
						break;
					}
					for (int l = j; l < j + scale; l++)
					{
						if ((l + scale) > img.cols)
						{
							break;
						}
						res = res | img.at<uchar>(Point(k, l));
					}
				}
				locs.x = i + (scale / 2); locs.y = j + (scale / 2);
				seed_pts.push_back(locs);
			}
		}
	}
	link1_clust(&v_points, seed_pts);

	for (vector<Pt>::iterator c = begin(seed_pts);
		c != end(seed_pts); ++c) {
		//quick hack to get clusterID
		int clusterId = c - begin(seed_pts);
		if (nPoints[clusterId] > 2)
		{	
			int minX = (c->x) - 2; int minY = (c->y) - 2; int maxX = (c->x) + 2; int maxY = (c->y) + 2;
			cv::rectangle(bnd_img, Point(minX,minY), Point(maxX, maxY), Scalar(0));
		}
	}
}

void Pose_Network::normalizergb(const cv::Mat& img, cv::Mat& edges)
{
	Mat bgr_image_f;
	img.convertTo(bgr_image_f, CV_32FC3);

	// Extract the color planes and calculate I = (r + g + b) / 3
	std::vector<cv::Mat> planes(3);
	split(bgr_image_f, planes);
	Mat intensity_f((planes[0] + planes[1] + planes[2]) / 3.0f);
	Mat b_normalized_f;
	divide(planes[0], intensity_f, b_normalized_f);
	Mat b_normalized;
	b_normalized_f.convertTo(b_normalized, CV_8UC1, 1.0);


	Mat g_normalized_f;
	divide(planes[1], intensity_f, g_normalized_f);
	Mat g_normalized;
	g_normalized_f.convertTo(g_normalized, CV_8UC1, 1.0);


	Mat r_normalized_f;
	divide(planes[2], intensity_f, r_normalized_f);
	Mat r_normalized;
	r_normalized_f.convertTo(r_normalized, CV_8UC1, 1.0);
	vector<Mat> ch;
	ch.push_back(b_normalized);
	ch.push_back(g_normalized);
	ch.push_back(r_normalized);
	merge(ch, edges);
}

bool Pose_Network::getpose(const Mat& in_img, const Mat& depth, cv::Vec3f& position) 
{
	if (!_pSession || !_pStatus || in_img.rows == 0 || in_img.cols == 0 || depth.rows == 0 || depth.cols == 0)
		return false;

	Rect myROI(0, 0, 380, 380);
	//Rect myROI(0, 0, 128, 128);
	//Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
	Mat img(in_img, myROI);
	Mat d(depth, myROI);

	Mat d_img;
	//imshow("img", img);

	Size size(64, 64);
//	Size size(128, 128);
	Mat edges1,edges;
	resize(d,d_img, size);
	resize(img, edges1, size);
	normalizergb(edges1, edges);
	//cv::normalize(edges1, edges, 0, 1, cv::NORM_MINMAX);
	imshow("edges1", edges1);
	moveWindow("edges1", 700, 200);
	//imshow("d_img",d_img);
	waitKey(1);
	//std::vector<int64_t> im_dims = { 1, loaded_img.rows, loaded_img.cols, 3 };
	//std::vector<float_t> im_data;
	// Load image into raw buffer
	std::vector<float_t> im_data;
	std::vector<int64_t> im_dims;
	loadImgData(edges, im_data, im_dims);

	std::vector<float_t> d_data;
	std::vector<int64_t> d_dims;
	//load1chData(d_img, d_data, d_dims);

    // Get the op from the graph to feed the input image into
    TF_Operation* input_op = getOp(_pGraph, "input_feed");
	TF_Operation* hidden_op = getOp(_pGraph, "keep_conv");
	std::vector<TF_Output> input_ops = {{input_op, 0},{hidden_op,0}};
	//const std::vector<float_t> hidden = { 1 };
	//const std::vector<int64_t> h_d = { 1 };
	//auto[h_dims, h_data] = std::make_tuple(h_d, hidden);
	std::vector<int64_t> dropout_dims = { 1 };
	std::vector<float_t> dropout_data = { 1 };

    // Create a tensor to hold the input image data
	//tf_utils::SetTensorData<float_t>(_pInputColorTensor, d_data);
	//tf_utils::SetTensorData<float_t>(_pDropoutTensor, hidden);

    TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT, im_dims, im_data);
	TF_Tensor* input_tensor2 = tf_utils::CreateTensor(TF_FLOAT, dropout_dims, dropout_data);
	std::vector<TF_Tensor*> input_tensors; // = { _pInputColorTensor, _pDropoutTensor };
	input_tensors.push_back(input_tensor);
	input_tensors.push_back(input_tensor2);

    // Get output op that holds number of detections in image
	auto op_seg = getOp(_pGraph, "Reshape_1");
	// output_tensors[0] will be allocated hold resulting tensor
	std::vector<TF_Tensor*> output_tensors = { nullptr };
	// Get output 0 from op_num_detections operation
	std::vector<TF_Output> output_ops = { {op_seg,0} };

	//Start Session to use graph to get the segmentation and pose
    auto sess_code = tf_utils::RunSession(_pSession, input_ops, input_tensors, output_ops, output_tensors, _pStatus);
	if (sess_code != TF_OK) 
	{
		deleteTensor(input_tensor);
		deleteTensor(input_tensor2);

		std::cout << "Failed to run session: " << TF_Message(_pStatus) << std::endl;
		return false;
	}

    // Print results of last run
    auto res = tf_utils::GetTensorData<float>(output_tensors[0]);
//	Mat pr = Mat(res).reshape(0, 128);
	Mat pr = Mat(res).reshape(0, 64);
	Mat bnd_img = pr;
	gen_bnd_box(pr, bnd_img, 10);
	imshow("bnd_img", bnd_img);
	moveWindow("bnd_img", 550, 200);
	waitKey(1);
	//Mat m1 = Mat(128, 128, CV_32F, 0.0);
	Mat m1 = Mat(64, 64, CV_32F, 0.0);
	cv::Mat diff = pr != m1;
	// Equal if no elements disagree
	bool eq = cv::countNonZero(diff) == 0;
	/*if (!eq)
	{
		count++;
		string name = "result" + std::to_string(count) + ".png";
		imwrite("D:\\captured\\" + name, diff);
		waitKey(1);
		string name1 = "rgb" + std::to_string(count) + ".png";
		imwrite("D:\\captured\\" + name1, edges1);
		waitKey(1);
	
	}*/
	// Stage 2 of the test
	//TF_Operation* in_op = getOp(_pGraph, "Placeholder");
	//TF_Operation* aug_op = getOp(_pGraph, "aug_map");
	//std::vector<float_t> in_data;
	//std::vector<int64_t> in_dims;
	//load1chData(pr, in_data, in_dims);

	//// Create a tensor to hold the input image data
	//TF_Tensor* input_s2tensor = tf_utils::CreateTensor(TF_FLOAT, in_dims, in_data);
	//TF_Tensor* input_s2tensor1 = tf_utils::CreateTensor(TF_FLOAT, d_dims, d_data);
	//std::vector<TF_Tensor*> input_tensors2;// = { _pInputSegmentedTensor, _pInputDepthTensor, _pDropoutTensor };
	//input_tensors2.push_back(input_s2tensor);
	//input_tensors2.push_back(input_s2tensor1);
	//input_tensors2.push_back(input_tensor2);
	////TF_Operation* hidden_op = getOp(_pGraph, "keep_hidden");
	//std::vector<TF_Output> in_ops = { {in_op, 0}, {aug_op,0},{hidden_op,0} };
	//
	//

	//// Get output op that holds number of detections in image
	//auto pose_t = getOp(_pGraph, "Add_1");
	//auto pose_q = getOp(_pGraph, "Add_3");
	//// output_tensors[0] will be allocated hold resulting tensor
	//std::vector<TF_Tensor*> output_tensors2 = { nullptr };
	//// Get output 0 from op_num_detections operation
	//std::vector<TF_Output> output_ops2 = { {pose_t,0},{pose_q,0} };

	//// Equivalent to sess.run(...)
	//sess_code = tf_utils::RunSession(_pSession, in_ops, input_tensors2, output_ops2, output_tensors2, _pStatus);
	//if (sess_code != TF_OK) 
	//{
	//	deleteTensor(input_tensor);
	//	deleteTensor(input_tensor2);
	//	deleteTensor(input_s2tensor);
	//	deleteTensor(input_s2tensor1);

	//	std::cout << "Failed to run session: " << TF_Message(_pStatus) << std::endl;
	//	return false;
	//}
	//// Print results of last run
	//auto res_1 = tf_utils::GetTensorData<float>(output_tensors2[0]);

	//position[0] = res_1[0];
	//position[1] = res_1[1];
	//position[2] = res_1[2];
	////position[3] = res_1[3];
	////position[4] = res_1[4];
	////position[5] = res_1[5];
	////position[6] = res_1[6];

	//deleteTensor(input_tensor);
	//deleteTensor(input_tensor2);
	//deleteTensor(input_s2tensor);
	//deleteTensor(input_s2tensor1);

    return true;
}



