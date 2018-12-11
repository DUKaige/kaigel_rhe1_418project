#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

#define kernel_width 7
#define threadsPerBlock 512


float stoff(const char* s){
  float rez = 0, fact = 1;
  if (*s == '-'){
    s++;
    fact = -1;
  };
  for (int point_seen = 0; *s; s++){
    if (*s == '.'){
      point_seen = 1; 
      continue;
    };
    int d = *s - '0';
    if (d >= 0 && d <= 9){
      if (point_seen) fact /= 10.0f;
      rez = rez * 10.0f + (float)d;
    };
  };
  return rez * fact;
}

__global__ void kernel_blur(float* pixels, float* output, int width, int height, int N) {
    const float kernel[kernel_width][kernel_width] = {
        {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
        {0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
        {0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
        {0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771},
        {0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
        {0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
        {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
    };

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int row = index / width;
        int col = index % width;
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
        output[index] = sum/denom;
    }   
}

__global__ void kernel_calculateGradient(float* pixelsAfterBlur, float* gradientMag, int* gradientAng, int width, int height, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int row = index / width;
        int col = index % width;
        float gy = (float) row == height - 1? 0 : pixelsAfterBlur[index + width] - pixelsAfterBlur[index];
        float gx = (float) col == width - 1? 0 : pixelsAfterBlur[index + 1] - pixelsAfterBlur[index];
        gradientMag[index] = sqrt(gx * gx + gy * gy);
        float ang;
        if (gx < 0.000001 && gx > -0.000001) ang = 90;
        else ang = atan(gy / gx) / 3.1415926 * 180.0;
        if (ang < 0)
            ang += 180;
        gradientAng[index] = ((int) (ang + 22.5) / 45) * 45;
    }
}

__global__ void kernel_doubleThreshold(float* pixelsAfterThin, int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height, float low_threshold, float high_threshold, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        float val = pixelsAfterThin[index];
        if (val >= high_threshold){
            pixelsStrongEdges[index] = 1;
        }
        if (val < high_threshold && val >= low_threshold){
            pixelsWeakEdges[index] = 1;
        }
    }
}

__global__ void kernel_thin(float* pixelsAfterThin, int* gradientAng, float* gradientMag, int width, int height, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int row = index / width;
        int col = index % width;
        float mag = gradientMag[row * width + col];
        float magL = 0;
        float magR = 0;
        int ang = gradientAng[row * width + col];
        if (ang == 0 || ang == 180) {
            if (row > 0) magL = gradientMag[row * width + col - 1];
            if (row < height - 1) magR = gradientMag[row * width + col + 1];
        } 

        else if (ang == 45 || ang == 225) {
            if (row > 0 && col < width - 1) magL = gradientMag[(row + 1) * width + col + 1];

            if (row < height - 1 && col > 0) magR = gradientMag[(row - 1) * width + col - 1];
        } 

        else if (ang == 90 || ang == 270) {
            if (col > 0) magL = gradientMag[(row - 1) * width + col];

            if (col < width - 1) magR = gradientMag[(row + 1) * width + col];
        } 

        else if (ang == 135 || ang == 315) {
            if (row > 0 && col > 0) magL = gradientMag[(row + 1) * width + col - 1];

            if (row < height - 1 && col < width - 1) magR = gradientMag[(row - 1) * width + col + 1];
        }
        if (mag > magL && mag > magR) {
            pixelsAfterThin[row * width + col] = mag;
        } 
        else {
            pixelsAfterThin[row * width + col] = 0;
        }
    }
}


void blur(float* pixels, float* output, int width, int height, int N, int blocks) {
    float* cudaPixels;
    float* cudaOutput;
    cudaMalloc(&cudaPixels, N * sizeof(float));
    cudaMalloc(&cudaOutput, N * sizeof(float));
    cudaMemcpy(cudaPixels, pixels, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel_blur<<<blocks, threadsPerBlock>>>(cudaPixels, cudaOutput, width, height, N);
    cudaThreadSynchronize();
    cudaMemcpy(output, cudaOutput, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);
    cudaFree(cudaOutput);
}

void calculateGradient(float* pixelsAfterBlur, float* gradientMag, int* gradientAng, int width, int height, float* maxMag, int N, int blocks) {
    float* cudaPixels;
    float* cudaGradientMag;
    int* cudaGradientAng;
    cudaMalloc(&cudaPixels, N * sizeof(float));
    cudaMalloc(&cudaGradientAng, N * sizeof(int));
    cudaMalloc(&cudaGradientMag, N * sizeof(float));
    cudaMemcpy(cudaPixels, pixelsAfterBlur, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel_calculateGradient<<<blocks, threadsPerBlock>>>(cudaPixels, cudaGradientMag, cudaGradientAng, width, height, N);
    cudaThreadSynchronize();
    cudaMemcpy(gradientMag, cudaGradientMag, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradientAng, cudaGradientAng, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);
    cudaFree(cudaGradientAng);
    cudaFree(cudaGradientMag);

    float max = 0;
    for (int i = 0; i < N; i++) {
        if (gradientMag[i] > max) {
            max = gradientMag[i];
        }
    }
    *maxMag = max;
}

void thin(float* gradientMag, int* gradientAng, float* pixelsAfterThin, int width, int height, int N, int blocks) {
    float* cudaPixels;
    float* cudaGradientMag;
    int* cudaGradientAng;
    cudaMalloc(&cudaPixels, N * sizeof(float));
    cudaMalloc(&cudaGradientAng, N * sizeof(int));
    cudaMalloc(&cudaGradientMag, N * sizeof(float));
    cudaMemcpy(cudaPixels, pixelsAfterThin, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaGradientAng, gradientAng, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaGradientMag, gradientMag, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel_thin<<<blocks, threadsPerBlock>>>(cudaPixels, cudaGradientAng, cudaGradientMag, width, height, N);
    cudaThreadSynchronize();
    cudaMemcpy(pixelsAfterThin, cudaPixels, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);
    cudaFree(cudaGradientMag);
    cudaFree(cudaGradientAng);
}

void doubleThreshold(float* pixelsAfterThin, int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height, float low_threshold, float high_threshold, int N, int blocks) {
    float* cudaPixels;
    int* cudaStrongEdges;
    int* cudaWeakEdges;
    cudaMalloc(&cudaPixels, N * sizeof(float));
    cudaMalloc(&cudaStrongEdges, N * sizeof(int));
    cudaMalloc(&cudaWeakEdges, N * sizeof(int));
    cudaMemcpy(cudaPixels, pixelsAfterThin, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel_doubleThreshold<<<blocks, threadsPerBlock>>>(cudaPixels, cudaStrongEdges, cudaWeakEdges, width, height, low_threshold, high_threshold, N);
    cudaThreadSynchronize();
    cudaMemcpy(pixelsStrongEdges, cudaStrongEdges, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pixelsWeakEdges, cudaWeakEdges, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);
    cudaFree(cudaStrongEdges);
    cudaFree(cudaWeakEdges);
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

float* split(string str, char delimiter, int numElts) {
    float* elts = (float*) malloc(sizeof(float) * numElts);
    stringstream ss(str);
    string tok;
    int i = 0; 

    while(getline(ss, tok, delimiter)) {
        elts[i++] = stoff(tok.c_str());
    }
 
    return elts;
}

int main(int argc, char** argv) {  

    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
            return -1;
    }

    float low_threshold = 0.05;
    float high_threshold = 0.1;
    float* pixels;
    int height;
    int width;

    string line;
    ifstream myfile (argv[1]);
    if (myfile.is_open()) {

        getline(myfile, line);
        height = stoff(line.c_str());
        getline(myfile, line);
        width = stoff(line.c_str());

        pixels = (float*) malloc(sizeof(float) * height * width);
        int idx = 0;
        while (getline(myfile, line)) {
            float* content = split(line, ' ', width);
            memcpy(pixels+idx, content, sizeof(float) * width);
            idx += width;
            free(content);
        }
        myfile.close();
    } 
    else {
        printf("Unable to open file"); 
        return -1;
    }

    int N = height * width;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // /* 1. blur */
    float* pixelsAfterBlur = (float*) malloc(sizeof(float)*height*width);
    blur(pixels, pixelsAfterBlur, width, height, N, blocks);

    /* 2. gradient */
    float* gradientMag = (float*) malloc(sizeof(float)*height*width);
    int* gradientAng = (int*) malloc(sizeof(int)*height*width);
    float maxMag = -1;
    calculateGradient(pixelsAfterBlur, gradientMag, gradientAng, width, height, &maxMag, N, blocks);

    /* 3. non-maximum suppresion */
    float* pixelsAfterThin = (float*) malloc(sizeof(float)*height*width);
    thin(gradientMag, gradientAng, pixelsAfterThin, width, height, N, blocks);

    /* 4. double thresholding */
    int* pixelsStrongEdges = (int*) calloc(sizeof(int), height*width);
    int* pixelsWeakEdges = (int*) calloc(sizeof(int), height*width);
    doubleThreshold(pixelsAfterThin, pixelsStrongEdges, pixelsWeakEdges, width, height, low_threshold * maxMag, high_threshold * maxMag, N, blocks);

    /* 5. edge tracking */
    edgeTrack(pixelsStrongEdges, pixelsWeakEdges, width, height);


    /* 6. display */
    ofstream outfile ("result.txt");
    if (outfile.is_open()) {

        outfile << height << "\n";
        outfile << width << "\n";
        int idx = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                outfile <<  pixelsStrongEdges[idx++] * 255 << " ";
            }
            outfile << "\n";
        }
        outfile.close();
    }

    return 0;
}
