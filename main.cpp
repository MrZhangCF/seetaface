#include "FaceRecognition.h"
#include "FeatureGroup.h"
#include <iostream>
#include <opencv2/opencv.hpp>

bool GetAllFileNames(string file_path, std::vector<string>& files){
	intptr_t hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(file_path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					GetAllFileNames(p.assign(file_path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				char *ext = strrchr(fileinfo.name, '.');
				if (ext){
					ext++;
					if (_stricmp(ext, "jpg") == 0 || _stricmp(ext, "png") == 0)
						files.push_back(p.assign(file_path).append("\\").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return true;
}


int main(int argc, char* argv[])
{
	cv::Mat frame;
	cv::VideoCapture capture(0);
	int choice=2;

	if (1 == choice)
	{
		std::vector<string> filenames;
		GetAllFileNames("./images", filenames);
		std::cout << "Detected " << filenames.size() << " images..." << std::endl;
		FaceRecognition fr;
		FeatureGroup fg(fr.GetFeatureDims(), &fr);
		float * feat = fr.NewFeatureBuffer();

		double averageExtractFeatureTime = 0.0;
		for (int i = 0; i < filenames.size(); i++)
		{
			int64 t0 = cv::getTickCount();
			if (fr.GetFeature(filenames[i], feat))
			{
				fg.AddFeature(feat, filenames[i]);
			}
			int64 t1 = cv::getTickCount();
			double secs = (t1 - t0) / cv::getTickFrequency();
			averageExtractFeatureTime += secs;
			if ((i + 1) % 5 == 0)
			{
				std::cout << i + 1 << " / " << int(filenames.size()) << "  has been extracted!"<< std::endl;
			}
		}
		std::cout << std::endl << "All verification takes " << averageExtractFeatureTime << " secs!" << std::endl;
		std::cout << "The average extract feature times for one image takes " << double(averageExtractFeatureTime / filenames.size() / 2) << " secs !" << std::endl;

		fg.SaveModel("feature.index");
		std::cout << "Feature Extraction has been finished!" << std::endl;
	}
	else if (2 == choice)
	{
		FaceRecognition fr;
		std::cout << "Loading DataBase..." << std::endl;
		FeatureGroup fg("feature.index", &fr);
		std::cout << "Database has been loaded!" << std::endl;
		while (true)
		{
			capture >> frame;

			Feature result;
			std::vector<float *> feats;
			ResultInfo ri;
			if (!feats.empty())
			{
				feats.clear();
			}

			//int facenum = fr.GetFeature(frame, feats);
			ri = fr.GetFeature(frame);
			int facenum = ri.facenum;

			std::cout << facenum << std::endl;

			for (std::vector<float*>::iterator it = feats.begin(); it != feats.end(); it++)
			{
				std::cout << *it << std::endl;
			}

			//std::cout << "break" << std::endl;
			for (int i = 0; i < facenum; i++)
			{
				result = fg.FindTop1(ri.feats[i]);
				std::cout << "No. " << i + 1 << " person is " << result.filename << " Similarity is " << result.similarity_with_goal << std::endl;
			}
			
			cv::imshow("frame", frame);
			cv::waitKey(1);
		}
		std::cout << std::endl;
	}

	return 0;
}