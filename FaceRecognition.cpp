#include "FaceRecognition.h"

Detector::Detector(const char* model_name) :seeta::FaceDetection(model_name)
{
	this->SetMinFaceSize(40);
	this->SetScoreThresh(2.f);
	this->SetImagePyramidScaleFactor(0.8f);
	this->SetWindowStep(4, 4);
}

FaceRecognition::FaceRecognition()
{
	this->detector = new Detector("model/seeta_fd_frontal_v1.0.bin");
	this->point_detector = new seeta::FaceAlignment("model/seeta_fa_v1.1.bin");
	this->face_recognizer = new seeta::FaceIdentification("model/seeta_fr_v1.0.bin");
}

float* FaceRecognition::NewFeatureBuffer()
{
	return new float[this->face_recognizer->feature_size()];
}

int FaceRecognition::GetFeatureDims()
{
	return this->face_recognizer->feature_size();
}

float FaceRecognition::FeatureCompare(float* feat1, float* feat2)
{
	return this->face_recognizer->CalcSimilarity(feat1, feat2);
}


bool FaceRecognition::GetFeature(string filename, float* feat)
{
	//load image and convert to gray
	cv::Mat src_img_color = cv::imread(filename, 1);
	cv::Mat src_img_gray;
	cv::cvtColor(src_img_color, src_img_gray, CV_BGR2GRAY);

	//convert to ImageData type
	seeta::ImageData src_img_data_color(src_img_color.cols, src_img_color.rows, src_img_color.channels());
	src_img_data_color.data = src_img_color.data;

	seeta::ImageData src_img_data_gray(src_img_gray.cols, src_img_gray.rows, src_img_gray.channels());
	src_img_data_gray.data = src_img_gray.data;

	//Detect faces
	std::vector<seeta::FaceInfo> faces = this->detector->Detect(src_img_data_gray);
	int32_t face_num = static_cast<int32_t>(faces.size());
	if (face_num == 0)
	{
		std::cout << "Faces are not detected." << std::endl;
		return false;
	}

	//Detect 5 facial landmarks
	seeta::FacialLandmark points[5];
	this->point_detector->PointDetectLandmarks(src_img_data_gray, faces[0], points);

	//Extract face identity feature
	this->face_recognizer->ExtractFeatureWithCrop(src_img_data_color, points, feat);

	return true;

}

ResultInfo FaceRecognition::GetFeature(cv::Mat& img)
{
	cv::Mat src_img_color = img;
	cv::Mat src_img_gray;
	cv::cvtColor(src_img_color, src_img_gray, CV_BGR2GRAY);

	float * feat = this->NewFeatureBuffer();

	ResultInfo ri;
	ri.facenum = 0;
	ri.feats.push_back(feat);

	//convert to ImageData type
	seeta::ImageData src_img_data_color(src_img_color.cols, src_img_color.rows, src_img_color.channels());
	src_img_data_color.data = src_img_color.data;

	seeta::ImageData src_img_data_gray(src_img_gray.cols, src_img_gray.rows, src_img_gray.channels());
	src_img_data_gray.data = src_img_gray.data;

	std::vector<seeta::FaceInfo> faces = this->detector->Detect(src_img_data_gray);
	int32_t face_num = static_cast<int32_t>(faces.size());
	if (face_num == 0)
	{
		std::cout << "Faces are not detected." << std::endl;
		return ri;
	}
	std::vector<float*> feats;
	if (!feats.empty())
	{
		feats.clear();
	}
	cv::Rect face_rect;
	for (int i = 0; i < face_num; i++)
	{
		cv::Rect face_rect;
		face_rect.x = faces[i].bbox.x;
		face_rect.y = faces[i].bbox.y;
		face_rect.width = faces[i].bbox.width;
		face_rect.height = faces[i].bbox.height;

		cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);

		float * feat = this->NewFeatureBuffer();
		//Detect 5 facial landmarks
		seeta::FacialLandmark points[5];
		this->point_detector->PointDetectLandmarks(src_img_data_gray, faces[i], points);

		//Extract face identity feature
		this->face_recognizer->ExtractFeatureWithCrop(src_img_data_color, points, feat);

		feats.push_back(feat);

	}
	ri.facenum = face_num;
	ri.feats.clear();
	ri.feats = feats;
	return ri;
}

FaceRecognition::~FaceRecognition()
{
	if (detector)
		delete detector;
	if (point_detector)
		delete point_detector;
	if (face_recognizer)
		delete face_recognizer;
}