#include <stdio.h>
#include <math.h> 

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>
#include <vector>
#include <chrono>

using namespace std;

#define kernel_width 7
#define NUMOFTHREADS 16
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
    // int leftBound = kernel_width/2;
    // int topBound = kernel_width/2;
    // int rightBound = width - kernel_width/2;
    // int botBound = height - kernel_width/2;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            //int border = 0;
            
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
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            int idx = row * width + col;
            float gy = (float) row == height - 1? 0 : pixelsAfterBlur[idx + width] - pixelsAfterBlur[idx];
            float gx = (float) col == width - 1? 0 : pixelsAfterBlur[idx + 1] - pixelsAfterBlur[idx];
            gradientMag[idx] = sqrt(gx * gx + gy * gy);
            
            float ang;
            if (gx < 0.000001 && gx > -0.000001) ang = 90;
            else ang = atan(gy / gx) / 3.1415926 * 180.0;
            if (ang < 0)
                ang += 180;

            //printf("(gy / gx: %f  ang: %f \n", gy / gx, ang);
            gradientAng[idx] = ((int) (ang + 22.5) / 45) * 45;
        }
    }

    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            int idx = row * width + col;
            if (gradientMag[idx] > *maxMag)
                *maxMag = gradientMag[idx];
        }
    }
}

void thin(float* gradientMag, int* gradientAng, float* pixelsAfterThin, int width, int height) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
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
            } else{
                pixelsAfterThin[row * width + col] = 0;
            }
        }
    }
}


void doubleThreshold(float* pixelsAfterThin, int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height, float low_threshold, float high_threshold) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < height; row ++) {
        for (int col = 0; col < width; col ++) {
            int idx = row * width + col;
            float val = pixelsAfterThin[idx];
            if (val >= high_threshold){
                pixelsStrongEdges[idx] = 1;
            }
            if (val < high_threshold && val >= low_threshold){
                pixelsWeakEdges[idx] = 1;
            }
        }
    }
}


void dfsRange(int row, int col, int lorow, int hirow, int locol, int hicol, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
    vector<int> stack;
    int idx = row * width + col;
    stack.push_back(idx);
    while (!stack.empty()) {
        idx = stack.back();
        stack.pop_back();
        if (pixelsWeakEdges[idx]) {
            pixelsStrongEdges[idx] = 1;
        }
        int id;
        if (pixelsStrongEdges[idx]) {
            if (row > lorow) {
                id = (row - 1) * width + col;
                if (!visited[id]) {
                    stack.push_back(id);
                    visited[id] = 1;
                }
                if (col > locol) {
                    id = (row - 1) * width + col - 1;
                    if (!visited[id]){
                        stack.push_back(id);
                        visited[id] = 1;
                    } 
                }

                if (col < hicol - 1) {
                    id = (row - 1) * width + col + 1;
                    if (!visited[id]) {
                        stack.push_back(id);
                        visited[id] = 1;
                    } 
                }
            }

            if (row < hirow - 1) {
                id = (row + 1) * width + col;
                if (!visited[id]) {
                    stack.push_back(id);
                    visited[id] = 1;
                }
                if (col > locol) {
                    id = (row + 1) * width + col - 1;
                    if (!visited[id]){
                        stack.push_back(id);
                        visited[id] = 1;
                    } 
                }

                if (col < hicol - 1) {
                    id = (row + 1) * width + col + 1;
                    if (!visited[id]) {
                        stack.push_back(id);
                        visited[id] = 1;
                    }
                }
            }

            if (col > locol) {
                id = row * width + col - 1;
                if (!visited[id]) {
                    stack.push_back(id);
                    visited[id] = 1;
                }            
            }

            if (col < hicol - 1) {
                id = row * width + col + 1;
                if (!visited[id]) {
                    stack.push_back(id);
                    visited[id] = 1;
                }               
            }
        }
    }    
}

void kernel_dfs(int index, int numDiv, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
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

void kernel_exchange(int index, int numDiv, int* pixelsStrongEdges, int* pixelsWeakEdges, int* visited, int width, int height) {
    int colIndex = index % numDiv;
    int rowIndex = index / numDiv;
    int colStart = colIndex * width / numDiv;
    int colEnd = (colIndex + 1) * width / numDiv;
    int rowStart = rowIndex * width / numDiv;
    int rowEnd = (rowIndex + 1) * width / numDiv;
    // Left
    if (colStart > 0) {
        for (int row = rowStart; row < rowEnd; row ++) {
            if (pixelsStrongEdges[row * width + colStart] == 1 && pixelsWeakEdges[row * width + colStart - 1] == 1) {
                pixelsStrongEdges[row * width + colStart - 1] = 1;
            }
        }
    }

    // Right
    if (colEnd < width) {
        for (int row = rowStart; row < rowEnd; row ++) {
            if (pixelsStrongEdges[row * width + colEnd] == 1 && pixelsWeakEdges[row * width + colStart + 1] == 1) {
                pixelsStrongEdges[row * width + colEnd + 1] = 1;
            }
        }
    }

    // Top
    if (rowStart > 0) {
        for (int col = colStart; col < colEnd; col ++) {
            if (pixelsStrongEdges[rowStart * width + col] == 1 && pixelsWeakEdges[(rowStart - 1) * width + colStart] == 1) {
                pixelsStrongEdges[(rowStart - 1) * width + col] = 1;
            }
        }
    }

    // Bottom
    if (rowEnd < height) {
        for (int col = colStart; col < colEnd; col ++) {
            if (pixelsStrongEdges[(rowEnd) * width + col] == 1 && pixelsWeakEdges[(rowStart + 1) * width + colStart] == 1) {
                pixelsStrongEdges[(rowEnd + 1) * width + col] = 1;
            }
        }
    }
}

void edgeTrack(int* pixelsStrongEdges, int* pixelsWeakEdges, int width, int height) {
    int* visited = (int*) calloc(sizeof(int), width * height);
    int numDiv = 16;
    for (int i = 0; i < 4; i += 1) {
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < numDiv*numDiv; index ++) {
            kernel_exchange(index, numDiv, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        }
        
        #pragma omp parallel for schedule(dynamic)
        for (int index = 0; index < numDiv*numDiv; index ++) {
            kernel_dfs(index, numDiv, pixelsStrongEdges, pixelsWeakEdges, visited, width, height);
        }
    }
}

float* split(string str, char delimiter, int numElts) {
    float* elts = (float*) malloc(sizeof(float) * numElts);
    stringstream ss(str);
    string tok;
    int i = 0; 

    while(getline(ss, tok, delimiter)) {
        elts[i++] = stof(tok);
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
        height = static_cast<int>(stof(line));
        getline(myfile, line);
        width = static_cast<int>(stof(line));

        pixels = (float*) malloc(sizeof(float) * height * width);
        int idx = 0;
        while (getline(myfile, line)) {
            float* content = split(line, ' ', width);
            for (int i = 0; i < width; i ++) {
                pixels[idx + i] = content[i];
            }
            idx += width;
            free(content);
        }
        myfile.close();
    } 
    else {
        printf("Unable to open file"); 
        return -1;
    }
    float* pixelsAfterBlur = (float*) malloc(sizeof(float)*height*width);
    float* gradientMag = (float*) malloc(sizeof(float)*height*width);
    int* gradientAng = (int*) malloc(sizeof(int)*height*width);
    float* pixelsAfterThin = (float*) malloc(sizeof(float)*height*width);
    int* pixelsStrongEdges = (int*) calloc(sizeof(int), height*width);
    int* pixelsWeakEdges = (int*) calloc(sizeof(int), height*width);

    printf("1\n");
    auto start = std::chrono::high_resolution_clock::now();
    /* 1. blur */
    blur(pixels, pixelsAfterBlur, width, height);
    printf("2\n");
    auto ck2 = std::chrono::high_resolution_clock::now();

    /* 2. gradient */

    float maxMag = -1;
    calculateGradient(pixelsAfterBlur, gradientMag, gradientAng, width, height, &maxMag);
    printf("3\n");
    auto ck3 = std::chrono::high_resolution_clock::now();

    /* 3. non-maximum suppresion */
    thin(gradientMag, gradientAng, pixelsAfterThin, width, height);
    printf("4\n");
    /* 4. double thresholding */
    auto ck4 = std::chrono::high_resolution_clock::now();

    doubleThreshold(pixelsAfterThin, pixelsStrongEdges, pixelsWeakEdges, width, height, low_threshold * maxMag, high_threshold * maxMag);
    printf("5\n");
    auto ck5 = std::chrono::high_resolution_clock::now();

    /* 5. edge tracking */
    edgeTrack(pixelsStrongEdges, pixelsWeakEdges, width, height);
    printf("6\n");
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
                outfile << pixelsStrongEdges[idx++] * 255 << " ";
            }
            outfile << "\n";
        }
        outfile.close();
    }

    return 0;
}
    