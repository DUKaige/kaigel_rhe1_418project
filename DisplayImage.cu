#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;

#define kernel_width 7
#define threadsPerBlock 512
#define NUM_ITER_DFS 3
#define DFS_BLOCK_SIZE 8
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    cudaMemcpy(gradientMag, cudaGradientMag, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradientAng, cudaGradientAng, N * sizeof(int), cudaMemcpyDeviceToHost);


    float max = 0;
    for (int i = 0; i < N; i++) {
        if (gradientMag[i] > max) {
            max = gradientMag[i];
        }
    }
    *maxMag = max;
    cudaFree(cudaPixels);
    cudaFree(cudaGradientAng);
    cudaFree(cudaGradientMag);
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    cudaMemcpy(pixelsStrongEdges, cudaStrongEdges, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pixelsWeakEdges, cudaWeakEdges, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);
    cudaFree(cudaStrongEdges);
    cudaFree(cudaWeakEdges);
}

__device__ __inline__ void push_back(int* stack, int* stack_pt, int val) {
    stack[*stack_pt] = val;
    *stack_pt ++;
}

__device__ __inline__ int empty(int* stack, int* stack_pt) {
    if (*stack_pt == 0) {
        return 1;
    } else {
        return 0;
    }
}

__device__ __inline__ int pop_back(int* stack, int* stack_pt) {
    int val = stack[*stack_pt - 1];
    *stack_pt --;
    return val;
}

__device__ __inline__ void dfsRange(int row, int col, int lorow, int hirow, int locol, int hicol, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
    int stack[DFS_BLOCK_SIZE * DFS_BLOCK_SIZE];
    int stack_pt = 0;
    int idx = row * width + col;
    push_back(stack, &stack_pt, idx);
    while (!empty(stack, &stack_pt)) {
        idx = pop_back(stack, &stack_pt);
        if (pixelsWeakEdges[idx]) {
            pixelsStrongEdges[idx] = 1;
        }
        int id;
        if (pixelsStrongEdges[idx]) {
            if (row > lorow) {
                id = (row - 1) * width + col;
                if (!visited[id]) {
                    push_back(stack, &stack_pt, id);
                    visited[id] = 1;
                }
                if (col > locol) {
                    id = (row - 1) * width + col - 1;
                    if (!visited[id]){
                        push_back(stack, &stack_pt, id);
                        visited[id] = 1;
                    } 
                }

                if (col < hicol - 1) {
                    id = (row - 1) * width + col + 1;
                    if (!visited[id]) {
                        push_back(stack, &stack_pt, id);
                        visited[id] = 1;
                    } 
                }
            }

            if (row < hirow - 1) {
                id = (row + 1) * width + col;
                if (!visited[id]) {
                    push_back(stack, &stack_pt, id);
                    visited[id] = 1;
                }
                if (col > locol) {
                    id = (row + 1) * width + col - 1;
                    if (!visited[id]){
                        push_back(stack, &stack_pt, id);
                        visited[id] = 1;
                    } 
                }

                if (col < hicol - 1) {
                    id = (row + 1) * width + col + 1;
                    if (!visited[id]) {
                        push_back(stack, &stack_pt, id);
                        visited[id] = 1;
                    }
                }
            }

            if (col > locol) {
                id = row * width + col - 1;
                if (!visited[id]) {
                    push_back(stack, &stack_pt, id);
                    visited[id] = 1;
                }            
            }

            if (col < hicol - 1) {
                id = row * width + col + 1;
                if (!visited[id]) {
                    push_back(stack, &stack_pt, id);
                    visited[id] = 1;
                }               
            }
        }
    }    
}

__global__ void kernel_dfs(int numDiv, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numDiv * numDiv) return;
    int colIndex = index % numDiv;
    int rowIndex = index / numDiv;
    int colStart = colIndex * width / numDiv;
    int colEnd = (colIndex + 1) * width / numDiv;
    int rowStart = rowIndex * width / numDiv;
    int rowEnd = (rowIndex + 1) * width / numDiv;


    for (int row = rowStart; row < rowEnd; row ++) {
        for (int col = colStart; col < colEnd; col ++) {
            if (pixelsStrongEdges[row * width + col] == 1)
                dfsRange(row, col, 0, height, 0, width, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);

        }
    }
}


__global__ void kernel_exchange(int numDiv, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numDiv * numDiv) return;
    int colIndex = index % numDiv;
    int rowIndex = index / numDiv;
    int colStart = colIndex * width / numDiv;
    int colEnd = (colIndex + 1) * width / numDiv;
    int rowStart = rowIndex * width / numDiv;
    int rowEnd = (rowIndex + 1) * width / numDiv;
    // Left
    if (colStart > 0) {
        for (int row = rowStart; row < rowEnd; row ++) {
            if (pixelsStrongEdges[row * width + colStart] == 1) {
                pixelsStrongEdges[row * width + colStart - 1] = 1;
            }
        }
    }

    // Right
    if (colEnd < width) {
        for (int row = rowStart; row < rowEnd; row ++) {
            if (pixelsStrongEdges[row * width + colEnd - 1] == 1) {
                pixelsStrongEdges[row * width + colEnd] = 1;
            }
        }
    }

    // Top
    if (rowStart > 0) {
        for (int col = colStart; col < colEnd; col ++) {
            if (pixelsStrongEdges[rowStart * width + col] == 1) {
                pixelsStrongEdges[(rowStart - 1) * width + col] = 1;
            }
        }
    }

    // Bottom
    if (rowEnd < height) {
        for (int col = colStart; col < colEnd; col ++) {
            if (pixelsStrongEdges[(rowEnd - 1) * width + col] == 1) {
                pixelsStrongEdges[rowEnd * width + col] = 1;
            }
        }
    }
}

void edgeTrack(int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height) {
    int* visited = (int*) calloc(sizeof(int), width * height);
    int numDiv = min((height + DFS_BLOCK_SIZE - 1)/DFS_BLOCK_SIZE
        , (width + DFS_BLOCK_SIZE - 1)/DFS_BLOCK_SIZE);
    int blocks = (numDiv * numDiv + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < NUM_ITER_DFS; i ++) {
        kernel_exchange<<<blocks, threadsPerBlock>>>(numDiv, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        cudaDeviceSynchronize();
        kernel_dfs<<<blocks, threadsPerBlock>>>(numDiv, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        cudaDeviceSynchronize();    
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

    float low_threshold = 0.1;
    float high_threshold = 0.15;
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
    auto start = std::chrono::high_resolution_clock::now();

    // /* 1. blur */
    float* pixelsAfterBlur = (float*) malloc(sizeof(float)*height*width);
    blur(pixels, pixelsAfterBlur, width, height, N, blocks);
    auto ck2 = std::chrono::high_resolution_clock::now();

    /* 2. gradient */
    float* gradientMag = (float*) malloc(sizeof(float)*height*width);
    int* gradientAng = (int*) malloc(sizeof(int)*height*width);
    float maxMag = -1;
    calculateGradient(pixelsAfterBlur, gradientMag, gradientAng, width, height, &maxMag, N, blocks);
    auto ck3 = std::chrono::high_resolution_clock::now();

    /* 3. non-maximum suppresion */
    float* pixelsAfterThin = (float*) malloc(sizeof(float)*height*width);
    thin(gradientMag, gradientAng, pixelsAfterThin, width, height, N, blocks);
    auto ck4 = std::chrono::high_resolution_clock::now();

    /* 4. double thresholding */
    int* pixelsStrongEdges = (int*) calloc(sizeof(int), height*width);
    int* pixelsWeakEdges = (int*) calloc(sizeof(int), height*width);
    doubleThreshold(pixelsAfterThin, pixelsStrongEdges, pixelsWeakEdges, width, height, low_threshold * maxMag, high_threshold * maxMag, N, blocks);
    auto ck5 = std::chrono::high_resolution_clock::now();

    /* 5. edge tracking */
    edgeTrack(pixelsStrongEdges, pixelsWeakEdges, width, height);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start);
    std::chrono::duration<double> blur_time = std::chrono::duration_cast<std::chrono::duration<double>>(ck2 - start);
    std::chrono::duration<double> grad_time = std::chrono::duration_cast<std::chrono::duration<double>>(ck3 - ck2);
    std::chrono::duration<double> sup_time = std::chrono::duration_cast<std::chrono::duration<double>>(ck4 - ck3);
    std::chrono::duration<double> db_ts = std::chrono::duration_cast<std::chrono::duration<double>>(ck5 - ck4);
    std::chrono::duration<double> ed_tk = std::chrono::duration_cast<std::chrono::duration<double>>(finish - ck5);
    std::cout << "Total: " << elapsed.count() << " seconds.\n";
    std::cout << "Blur: " << blur_time.count() << " seconds.\n";
    std::cout << "Gradient: " << grad_time.count() << " seconds.\n";
    std::cout << "Non-max sup: " << sup_time.count() << " seconds.\n";
    std::cout << "Double thresholding: " << db_ts.count() << " seconds.\n";
    std::cout << "Edge tracking: " << ed_tk.count() << " seconds.\n";

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