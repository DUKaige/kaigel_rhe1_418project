#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

#define JUMP 10

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
struct coordinate {
	int x;
	int y;
};

int acute(int prev_x, int prev_y, int current_x, int current_y, int next_x, int next_y) {
	if (prev_x == -1) return 1;
	int prod = (current_x - prev_x) * (next_x - current_x) 
		+ (current_y - prev_y) * (next_y - current_y);
	if (prod > 0) {
		return 1;
	} else {
		return 0;
	}
}

// Return 1 if there is a path
int dfs(int init_x, int init_y, int current_x, int current_y, 
	int* visited, int* parents, float* pixels, int height, int width,int max_jump) {

	vector<struct coordinate> stack;
	struct coordinate init_coor;
	init_coor.x = init_x;
	init_coor.y = init_y;
	stack.push_back(init_coor);
	int times_passed_init = 0;

	while (!stack.empty()) {
		struct coordinate cur = stack.back();
		stack.pop_back();
		int current_x = cur.x;
		int current_y = cur.y;
		printf("x: %d y: %d\n", current_x, current_y);
		if (current_x == init_x && current_y == init_y && times_passed_init > 0) {
			return 1;
		}
		if (times_passed_init == 1)
			visited[current_y * width + current_x] = 1;
		times_passed_init = 1;

		int flag = 0;

		for (int next_x = max(current_x - 1, 0); next_x < min(current_x + 2, width); next_x ++) {
			for (int next_y = max(current_y - 1, 0); next_y < min(current_y + 2, height); next_y ++) {
				int id = next_y * width + next_x;
				if (pixels[id] != 0 && !visited[id]) {
					flag = 1;
					parents[id] = current_y * width + current_x;
					struct coordinate next;
					next.x = next_x;
					next.y = next_y;
					stack.push_back(next);
				}
			}
		}

		if (flag == 0) {
			int prev_x = -1;
			int prev_y = -1;
			int parent = parents[current_y * width + current_x];
			if (parent > 0) {
				prev_x = parent%width;
				prev_y = parent/width;
			}

			for (int next_x = max(current_x - max_jump, 0); next_x < min(current_x + max_jump + 1, width); next_x ++) {
				for (int next_y = max(current_y - max_jump, 0); next_y < min(current_y + max_jump + 1, height); next_y ++) {
					int id = next_y * width + next_x;

					if (acute(prev_x, prev_y, cur.x, cur.y, next_x, next_y) 
						&& (next_x - current_x) *  (next_x - current_x) + (next_y - current_y) *  (next_y - current_y) >= 2
						&& pixels[id] != 0 
						&& !visited[id]) {
						parents[id] = current_y * width + current_x;
						struct coordinate next;
						next.x = next_x;
						next.y = next_y;
						stack.push_back(next);
					}
				}
			}
		}
	}
	return 0;
}

//Return 0 if no result
//Return 1 if there is result, and will directly draw to mat.
int explore_angle(int angle, float* pixels, float* mat, int height, int width, 
	int starting_x, int starting_y) {
	int init_x;
	int init_y;
	if (angle == 0) {
		for (int x = starting_x; x < width; x += 1) {
			if (pixels[starting_y * width + x] == 1){
				init_x = x;
				init_y = starting_y;
				break;
			} 
		}
	} 
	//TODO: add more angles.

	int* visited = (int*)calloc(sizeof(int), width * height);
	int* parents = (int*)calloc(sizeof(int), width * height);;
	int result = dfs(init_x, init_y, init_x, init_y, 
							visited, parents, pixels, height, width, JUMP);
	free(visited);

	if (result == 0) {
		printf("non\n");
		return 0;
	} else {
		int cur_x = init_x;
		int cur_y = init_y;
		int times_passed_init = 0;
		while (cur_x != init_x || cur_y != init_y || times_passed_init == 0) {
			times_passed_init = 1;
			int id = cur_y * width + cur_x;
			mat[id] = 1;
			printf("path x: %d y: %d\n", cur_x, cur_y);
			cur_x = parents[id] % width;
			cur_y = parents[id] / width;
		}
		return 1;
	}
}

int main(int argc, char** argv) {  
  if (argc != 4) {
      printf("usage: r/l <Image_Path> x_coor y_coor\n");
          return -1;
  }

  int remove = 1;
  if (argv[0][0] == 'l') {
  	remove = 0;
  }

	float* pixels;
	int starting_x = atoi(argv[2]);
	int starting_y = atoi(argv[3]);
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
  for (int y = 0; y < height; y++) {
	  for (int x = 0; x < width; x ++) {
	  	int id = y * width + x;
	  	pixels[id] = pixels[id] == 0 ? 0 : 1;
	  }
  }
  //Draw edges on edge of the image.
  /*for (int x = 0; x < width; x ++) {
  	pixels[x] = 1;
  	pixels[(height - 1) * width + x] = 1;
  }*/

  for (int y = 0; y < height; y ++) {
  	pixels[y * width] = 1;
  	pixels[y * width + width - 1] = 1;
  }

  
	float* mat = (float*)calloc(sizeof(float), height * width);
  explore_angle(0, pixels, mat, height, width, 
	starting_x, starting_y);


  ofstream outfile ("seg_result.txt");
  if (outfile.is_open()) {

      outfile << height << "\n";
      outfile << width << "\n";
      int idx = 0;
      for (int i = 0; i < height; i++) {
          for (int j = 0; j < width; j++) {
              outfile <<  mat[idx++] * 255 << " ";
          }
          outfile << "\n";
      }
      outfile.close();
  }

  return 0;
}