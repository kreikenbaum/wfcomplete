#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MIN 1
#define SUM 2


int get_index(int web, int trial, int TRIALS){
	return (web-1)*TRIALS+trial -1;
}

inline int min(int a, int b){
	return a < b ? a : b;
}

int parse(char* fname, double aug){
	FILE* fp;
	int count,length;
	if((fp = fopen(fname, "r")) == NULL){
		printf("%s cannot be opened!\n", fname);
		exit(1);
	}

	count = 0;
	while(!feof(fp)){
		if(0 > fscanf(fp, "%d", &length))
			continue;
		if(abs(length) == 52 || abs(length) == 40)
			continue;
		count++;
	}
	
	fclose(fp);
	return count == 0 ? 1 : (int)ceil(count*aug);
}

void gen_gamma_matrix(int power, char* fmatrix, char* gamma_matrix, int method, int WEBSITES, int TRIALS){
	int index, i, j, web, trial, strial;
	int dim = WEBSITES*TRIALS;

	int* size = new int[WEBSITES*TRIALS];

	FILE* fin, *fout;
	char fname[200];
	double dis;
	double gamma = pow(2, power);

	//write to gamma matrix
	if((fin = fopen(fmatrix, "r")) == NULL){
		printf("%s cannot be opened!\n", fmatrix);
		exit(1);
	}

	if((fout = fopen(gamma_matrix, "w")) == NULL){
		printf("%s cannot be opened!\n", gamma_matrix);
		fclose(fin);
		exit(1);
	}

	for(i = 0; i < dim; i++){
		for(j = 0; j < dim; j++){
			fscanf(fin, "%lf", &dis);
			//apply gamma
			dis = exp(-gamma*dis*dis);
			fprintf(fout, "%E ", dis);
		}
		fprintf(fout, "\n");
	}
	delete[] size;

	fclose(fout);
	fclose(fin);
}


int main(int argc, char** argv){
	if(argc != 4){
		printf("./gen_gamma_matrix <pow> <folder> <input_matrix> <output_matrix>\n");
		exit(1);
	}
	int WEBSITES = 100;
	int TRIALS = 40;
	gen_gamma_matrix(atoi(argv[1]),argv[2], argv[3], MIN, WEBSITES, TRIALS);
	return 0;
}
