#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
float* split(string str, char delimiter, int numElts) {
    float* elts = (float*) malloc(sizeof(float) * numElts);
    stringstream ss(str);
    string tok;
    int i = 0; 

    while(getline(ss, tok, delimiter)) {
        elts[i++] = stoi(tok, nullptr);
    }
 
    return elts;
}

int main(int argc, char** argv) {  
		if (argc != 4) {
		    printf("argc: %d\n", argc);
		    return -1;
	  }
	  char mode = argv[1][0];
	  char* img = argv[2];
	  char* txt = argv[3];

	  if (mode == 't') {
				Mat image;
		    image = imread(img, 0);
		    if (!image.data) {
		        printf("No image data \n");
		        return -1;
		    }
		    int width = image.cols;
		    int height = image.rows;

				ofstream file;
		    file.open (txt);
		    file << to_string(height) << "\n";
		    file << to_string(width) << "\n";
				for (int row = 0; row < height; row ++) {
		        for (int col = 0; col < width; col ++) {
			          Scalar intensity = image.at<uchar>(row, col);
			          float pixel = intensity.val[0];
			          file << to_string(pixel);
			          file << ' ';
		        }
		        file << "\n";
		    }
		    file.close();
	  } else {
	  		string line;
				ifstream file(txt);
	  		getline(file, line);
	  		cout << "l1" << line << endl;
	  		int height = stoi(line);

	  		getline(file, line);
	  		cout << "l2" << line << endl;

	  		int width = stoi(line);
				
				Mat mat(height, width, CV_8UC4);

	  		for (int row = 0; row < height; row ++) {
	  				getline(file, line);
		  			float* splitted = split(line, ' ', width);
		  			for (int col = 0; col < width; col ++) {
			  				Vec4b& rgba = mat.at<Vec4b>(row, col);
		            int val = splitted[col];
		            rgba[0] = val;
		            rgba[1] = val;
		            rgba[2] = val;
		            rgba[3] = 255;
		  			}
		  			free(splitted);
	  		}

	  		vector<int> compression_params;
		    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		    compression_params.push_back(95);

		    try {
        		imwrite(img, mat, compression_params);
		    }
		    catch (runtime_error& ex) {
		        fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
		        return 1;
		    }
	  }
	  return 0;
}