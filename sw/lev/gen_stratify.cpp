#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

int WEBSITES = 100;
int TRIALS = 40;

void parse(char* fmatrix){

	int trainsize = int(TRIALS * 0.9);

	int label,id;
	int dim = WEBSITES * TRIALS;
	int instances = TRIALS/10;
	double cdis;
	FILE* fin;
	FILE** fout_train, **fout_test;
	char outname[100];
	double* tmp = (double*)malloc(dim*sizeof(double));

	
	if((fin = fopen(fmatrix,"r")) == NULL){
		cout<<"cannot open file "<<fmatrix<<endl;
		exit(1);
	}
	printf("%d \n", 0);

	fout_train = (FILE**)malloc(10*sizeof(FILE*));
	fout_test = (FILE**)malloc(10*sizeof(FILE*));

	for(int i = 0; i < 10; i++){
		memset(outname, 0, 100);
		sprintf(outname, "./gnorm_cus_testing_%d", i+1);
		if((fout_test[i] = fopen(outname,"a+")) == NULL){
			cout<<"cannot open file "<<outname<<endl;
			exit(1);
		}
		memset(outname, 0, 100);
		sprintf(outname, "./gnorm_cus_training_%d", i+1);
		if((fout_train[i] = fopen(outname,"a+")) == NULL){
			cout<<"cannot open file "<<outname<<endl;
			exit(1);
		}
	}

	int testfold;

	for(int web = 1; web <= WEBSITES; web++){
		for(int trial = 1; trial <= TRIALS; trial++){
			//read the line
			for(int i = 0; i < dim; i++)
				fscanf(fin, "%lf", &(tmp[i]));

			testfold = (trial-1)/instances+1;
			for(int fold = 1; fold <= 10; fold++){
				if(fold == testfold){
					fprintf(fout_test[fold-1],"%d 0:%d ", web, (web-1)*TRIALS+trial);
					for(int i = 1; i <= dim; i++){
						fprintf(fout_test[fold-1], "%d:%lf ",i,tmp[i-1]);
					}
					fprintf(fout_test[fold-1], "\n");
				}
				else{
					fprintf(fout_train[fold-1],"%d 0:%d ", web, (web-1)*TRIALS+trial);
					for(int i = 1; i <= dim; i++){
						fprintf(fout_train[fold-1], "%d:%lf ",i,tmp[i-1]);
					}
					fprintf(fout_train[fold-1], "\n");
				}
			}
		}
	}

	for(int i = 0; i < 10; i++){
		fclose(fout_test[i]);
		fclose(fout_train[i]);
	}
	free(fout_test);
	free(fout_train);
	fclose(fin);
	free(tmp);
}


int main(int argc, char** argv){
	if(argc != 4){
		cout<<"usage: ./gen_stratify <#websites> <#trials> <matrix>"<<endl;
		exit(1);
	}
	WEBSITES = atoi(argv[1]);
	TRIALS = atoi(argv[2]);
	char* matrix = argv[3];
	parse(matrix);
	return 0;
}


