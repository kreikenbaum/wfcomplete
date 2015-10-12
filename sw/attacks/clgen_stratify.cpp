#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

void parse(int websites, int trainsize, int trials, char* fmatrix){

	int label,id;
	int dim = websites*trials;
	int buffersize = websites*trials;
	double cdis;
	FILE* fin;
	FILE* fout_train, *fout_test;
	char outname[100];
	double* tmp = (double*)malloc(buffersize*sizeof(double));

	
	if((fin = fopen(fmatrix,"r")) == NULL){
		cout<<"cannot open file "<<fmatrix<<endl;
		exit(1);
	}
	printf("%d \n", 0);

	memset(outname, 0, 100);
	sprintf(outname, "./cl.test");
	if((fout_test = fopen(outname,"a+")) == NULL){
		cout<<"cannot open file "<<outname<<endl;
		exit(1);
	}
	memset(outname, 0, 100);
	sprintf(outname, "./cl.train");
	if((fout_train = fopen(outname,"a+")) == NULL){
		cout<<"cannot open file "<<outname<<endl;
		exit(1);
	}

	for(int web = 1; web <= websites; web++){
		for(int trial = 1; trial <= trials; trial++){
			//read the line
			for(int i = 0; i < buffersize; i++)
				fscanf(fin, "%lf", &(tmp[i]));

			if(trial <= trainsize){
				fprintf(fout_test,"%d 0:%d ", web, (web-1)*trials+trial);
				for(int i = 1; i <= dim; i++){
					fprintf(fout_test, "%d:%lf ",i,tmp[i-1]);
				}
				fprintf(fout_test, "\n");
			}
			else{
				fprintf(fout_train,"%d 0:%d ", web, (web-1)*trials+trial);
				for(int i = 1; i <= dim; i++){
					fprintf(fout_train, "%d:%lf ",i,tmp[i-1]);
				}
				fprintf(fout_train, "\n");
			}
		}
	}
	fclose(fout_test);
	fclose(fout_train);
	fclose(fin);
	free(tmp);
}


int main(int argc, char** argv){
	if(argc != 4){
		cout<<"usage: ./clgen_stratify <matrixname> <trainsize> <train+testsize>"<<endl;
		cout<<"example: ./clgen_stratify ./tor_100_40_gnorm_matrix 5 10"<<endl;
		exit(1);
	}
	int websites = 100;
	char* matrix = argv[1]; //file name of matrix. tor_100_40_gnorm_matrix
	int trainsize = atoi(argv[2]); //training size. 
	int trials = atoi(argv[3]); //training size. 
	parse(websites, trainsize, trials, matrix);
	return 0;
}


