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

void calculateGradient(float* pixelsAfterBlur, float* gradientMag, int* gradientAng, int width, int height, float* maxMag) {
    int idx = 0;
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            float gy = (float) row == height - 1? 0 : pixelsAfterBlur[idx + width] - pixelsAfterBlur[idx];
            float gx = (float) col == width - 1? 0 : pixelsAfterBlur[idx + 1] - pixelsAfterBlur[idx];
            gradientMag[idx] = sqrt(gx * gx + gy * gy);
            if (gradientMag[idx] > *maxMag)
                *maxMag = gradientMag[idx];
            float ang;
            if (gx < 0.000001 && gx > -0.000001) ang = 90;
            else ang = atan(gy / gx) / 3.1415926 * 180.0;
            if (ang < 0)
                ang += 360;
            printf("(gy / gx: %f  ang: %f \n", gy / gx, ang);
            gradientAng[idx] = ((int) (ang + 22.5) / 45) * 45;
            idx++; 
        }
    }
}

void thin(float* gradientMag, int* gradientAng, float* pixelsAfterThin, int width, int height) {
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            float mag = gradientMag[row * width + col];
            float magL = 0;
            float magR = 0;
            int ang = gradientAng[row * width + col];
            if (ang == 0 || ang == 180) {
                if (row > 0) magL = gradientMag[(row - 1) * width + col];

                if (row < height - 1) magR = gradientMag[(row + 1) * width + col];
            } 

            else if (ang == 45 || ang == 225) {
                if (row > 0 && col < width - 1) magL = gradientMag[(row - 1) * width + col + 1];

                if (row < height - 1 && col > 0) magR = gradientMag[(row + 1) * width + col - 1];
            } 

            else if (ang == 90 || ang == 270) {
                if (col > 0) magL = gradientMag[row * width + col - 1];

                if (col < width - 1) magR = gradientMag[row * width + col + 1];
            } 

            else if (ang == 135 || ang == 315) {
                if (row > 0 && col > 0) magL = gradientMag[(row - 1) * width + col - 1];

                if (row < height - 1 && col < width - 1) magR = gradientMag[(row + 1) * width + col + 1];
            }
            if (mag > magL && mag > magR) {
                pixelsAfterThin[row * width + col] = mag;
            } else{
                pixelsAfterThin[row * width + col] = 0;
            }
        }
    }
}


void doubleThreshold(float* pixelsAfterThin, int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height, float low_threshold, float high_threshold) {
    int idx = 0;
    printf("%f %f\n", low_threshold, high_threshold);
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            float val = pixelsAfterThin[idx];
            if (val >= high_threshold){
                pixelsStrongEdges[idx] = 1;
            }
            if (val < high_threshold && val >= low_threshold){
                pixelsWeakEdges[idx] = 1;
            }
            idx++;
        }
    }
}

void dfs(int row, int col, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
    if (row < 0 || row >= height || col < 0 || col >= width)
        return;
    int idx = row * width + col;
    if (visited[idx] == 1){
        return;
    }
    if (pixelsWeakEdges[idx]) {
        pixelsStrongEdges[idx] = 1;
    }
    else if (!pixelsStrongEdges[idx]) {
        return;
    }

    visited[idx] = 1;
    if (row > 0) {
        dfs(row - 1, col, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        if (col > 0) {
            dfs(row - 1, col - 1, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        }

        if (col < width - 1) {
            dfs(row - 1, col + 1, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        }
    }

    if (row < height - 1) {
        dfs(row + 1, col, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        if (col > 0) {
            dfs(row + 1, col - 1, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        }
        if (col < width - 1) {
            dfs(row + 1, col + 1, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        }
    }

    if (col > 0) {
        dfs(row, col - 1, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
    }

    if (col < width - 1) {
        dfs(row, col + 1, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
    }
}

void edgeTrack(int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height) {
    int* visited = (int*) calloc(sizeof(int), width * height);
    int idx = 0;
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            if (pixelsStrongEdges[idx] == 1)
                dfs(row, col, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
            idx++;
        }
    }
}


int main(int argc, char** argv) {    
    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
            return -1;
    }

    float low_threshold = 0.2;
    float high_threshold = 0.6;

    Mat image;
    image = imread(argv[1], 0);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    int width = image.cols;
    int height = image.rows;
    float* pixels = (float*) malloc(sizeof(float)*height*width);
    int idx = 0;
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            Scalar intensity = image.at<uchar>(row, col);
            pixels[idx] = intensity.val[0];
            idx++;
        }
    }

    /* 1. blur */
    float* pixelsAfterBlur = (float*) malloc(sizeof(float)*height*width);
    blur(pixels, pixelsAfterBlur, width, height);

    /* 2. gradient */
    float* gradientMag = (float*) malloc(sizeof(float)*height*width);
    int* gradientAng = (int*) malloc(sizeof(int)*height*width);
    float maxMag = -1;
    calculateGradient(pixelsAfterBlur, gradientMag, gradientAng, width, height, &maxMag);

    /* 3. non-maximum suppresion */
    float* pixelsAfterThin = (float*) malloc(sizeof(float)*height*width);
    thin(gradientMag, gradientAng, pixelsAfterThin, width, height);

    /* 4. double thresholding */
    int* pixelsStrongEdges = (int*) calloc(sizeof(int), height*width);
    int* pixelsWeakEdges = (int*) calloc(sizeof(int), height*width);
    doubleThreshold(pixelsAfterThin, pixelsStrongEdges, pixelsWeakEdges, width, height, low_threshold * maxMag, high_threshold * maxMag);

    /* 5. edge tracking */
    edgeTrack(pixelsStrongEdges, pixelsWeakEdges, width, height);


    /* 6. display */
    Mat mat(height, width, CV_8UC4);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            Vec4b& rgba = mat.at<Vec4b>(i, j);
            int val = pixelsAfterThin[i * width + j];
            rgba[0] = val;
            rgba[1] = val;
            rgba[2] = val;
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
