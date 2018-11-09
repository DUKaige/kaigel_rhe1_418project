#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
#define kernel_width 7
const float kernel[kernel_width][kernel_width] = {
    {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
    {0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
    {0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
    {0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771},
    {0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
    {0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
    {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
};

void blur(float* pixels, float* output, int width, int height) {
    int leftBound = kernel_width/2;
    int topBound = kernel_width/2;
    int rightBound = width - kernel_width/2;
    int botBound = height - kernel_width/2;
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            int border = 0;
            
            float sum = 0;
            float denom = 0;
            int rowStart = row - kernel_width/2;
            int rowEnd = row + kernel_width/2 + 1;
            int colStart = col - kernel_width/2;
            int colEnd = col + kernel_width/2 + 1;
            for (int smallRow = rowStart; smallRow < rowEnd; smallRow ++) {
                for (int smallCol = colStart; smallCol < colEnd; smallCol ++) {
                    if (smallRow >= 0 && smallRow < height && smallCol >= 0 && smallCol < width) {
                        sum += kernel[smallRow - rowStart][smallCol - colStart] * pixels[smallRow * width + smallCol];
                        denom += kernel[smallRow - rowStart][smallCol - colStart];
                    }
                }
            }
            output[row * width + col] = sum/denom;

        }
    }
}

void calculateGradient(float* pixelsAfterBlur, float* gradientMag, float* gradientAng, int width, int height) {

}

void thin(float* gradientMag, float* gradientAng, float* pixelsAfterThin, int width, int height) {

}

void doubleThreshold(float* pixelsAfterThin, float* pixelsStrongEdges, float* pixelsWeakEdges, int width, int height) {

}

void edgeTrack(float* pixelsStrongEdges, float* pixelsWeakEdges, float* pixelsAfterTrack, int width, int height) {

}


int main(int argc, char** argv) {    
    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
            return -1;
    }

    Mat image;
    image = imread(argv[1], 0);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    int width = image.cols;
    int height = image.rows;
    float* pixels = (float*) malloc(sizeof(float)*height*width);
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            Scalar intensity = image.at<uchar>(row, col);
            pixels[row * width + col] = intensity.val[0];
        }
    }

    /* 1. blur */
    float* pixelsAfterBlur = (float*) malloc(sizeof(float)*height*width);
    blur(pixels, pixelsAfterBlur, width, height);

    /* 2. gradient */
    float* gradientMag = (float*) malloc(sizeof(float)*height*width);
    float* gradientAng = (float*) malloc(sizeof(float)*height*width);
    calculateGradient(pixelsAfterBlur, gradientMag, gradientAng, width, height);

    /* 3. non-maximum suppresion */
    float* pixelsAfterThin = (float*) malloc(sizeof(float)*height*width);
    thin(gradientMag, gradientAng, pixelsAfterThin, width, height);

    /* 4. double thresholding */
    float* pixelsStrongEdges = (float*) malloc(sizeof(float)*height*width);
    float* pixelsWeakEdges = (float*) malloc(sizeof(float)*height*width);
    doubleThreshold(pixelsAfterThin, pixelsStrongEdges, pixelsWeakEdges, width, height);

    /* 5. edge tracking */
    float* pixelsAfterTrack = (float*) malloc(sizeof(float)*height*width);
    edgeTrack(pixelsStrongEdges, pixelsWeakEdges, pixelsAfterTrack, width, height);


    /* 6. display */
    Mat mat(height, width, CV_8UC4);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            Vec4b& rgba = mat.at<Vec4b>(i, j);
            rgba[0] = pixelsAfterBlur[i * width + j];
            rgba[1] = pixelsAfterBlur[i * width + j];
            rgba[2] = pixelsAfterBlur[i * width + j];
            rgba[3] = 255;
        }
    }

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    try {
        imwrite("wow.jpg", mat, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
        return 1;
    }


    imshow("Display Image", mat);

    waitKey(0);
    return 0;
}
