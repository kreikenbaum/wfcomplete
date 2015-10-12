#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int WEBSITES = 100;
int TRIALS = 40;

int get_index(int web, int trial){
	return (web-1)*TRIALS+trial-1;
}

void parse_onefile(double** matrix, char* infname){
	int webx,weby,trialx,trialy,lengthx,lengthy;
	double dis;
	char cruft;

	int line = 0;
	FILE* fin;
	if((fin = fopen(infname, "r")) == NULL){
		printf("cannot open %s for reading\n", infname);
		exit(1);
	}
	while(fscanf(fin, "%d;%d;%d;%d;%d;%d;%lf%c", &webx,&trialx,&weby,&trialy,&lengthx,&lengthy,&dis,&cruft) == 8){
		dis = exp(-2*dis*dis); //merging gen_gamma_matrix
		matrix[get_index(webx,trialx)][get_index(weby,trialy)] = dis;
		line ++;
	}
	printf("parsing %s finished, lines = %d\n",infname,line);
	fclose(fin);
}


void writetofile(double** matrix, char* outfname){
	int dim = WEBSITES*TRIALS;
	int i, j;
	
	FILE* fout;
	if((fout = fopen(outfname, "w")) == NULL){
		printf("cannot open %s for writing\n", outfname);
		exit(1);
	}
	
	for(i = 0; i < dim; i++){
		for(j = 0; j < dim; j++){
			fprintf(fout, "%lf ", matrix[i][j]);
		}
		fprintf(fout, "\n");
	}
}

int main(int argc, char** argv){


	if(argc < 5){
	    printf("example: ./cantor2matrix <output> <#websites> <#trials> <files...\n");
	    exit(1);
	}

	char* outfname = argv[1];
	WEBSITES = atoi(argv[2]); 
	TRIALS = atoi(argv[3]);

	int i;
	int dim = WEBSITES*TRIALS;

	double** matrix = (double**)malloc(sizeof(double*) * dim);
	for(i = 0; i < dim; i++){
		matrix[i] = (double*)malloc(sizeof(double)* dim);
	}
	for(i = 0; i < dim; i++)
		matrix[i][i] = 0.0f;
	
	for(i = 4; i < argc; i++){
		parse_onefile(matrix, argv[i]);
	}
	printf("start writing to file ...\n");
	writetofile(matrix, outfname);
	printf("finished\n");
	return 0;
}

