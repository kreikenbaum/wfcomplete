#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <string.h>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
using namespace std;



//Data parameters
int FEAT_NUM = 3736; //number of features

string folder = "../padding/defdata2/"; //folder with elements for distance learning
string folderopen = ""; //folder with open elements

//foldertest is different from foldertrain when simulating an attacker with imperfect information
string foldertrain = "../padding/defdata2/"; //folder that contains training elements (knn background)
string foldertest = "../padding/defdata2/"; //folder that contains testing elements

const int SITE_NUM = 100; //number of monitored sites
const int INST_NUM = 40; //number of instances per site for distance learning
int TEST_NUM = 40; //number of instances per site for kNN training/testing
int OPENTEST_NUM = 0; //number of open instances for kNN training/testing
int NEIGHBOUR_NUM = 5; //number of neighbors for kNN
int RECOPOINTS_NUM = 5; //number of neighbors for distance learning

//Algorithmic Parameters
//float POWER = 0.1; //not used in this code; check float dist()


bool inarray(int ele, int* array, int len) {
	for (int i = 0; i < len; i++) {
		if (array[i] == ele)
			return 1;
	}
	return 0;
}

void alg_init_weight(float** feat, float* weight) {
	for (int i = 0; i < FEAT_NUM; i++) {
		weight[i] = (rand() % 100) / 100.0 + 0.5;
	}
	/*float sum = 0;
	for (int j = 0; j < FEAT_NUM; j++) {
		if (abs(weight[j]) > sum) {
		sum += abs(weight[j]);
		}
	}
	for (int j = 0; j < FEAT_NUM; j++) {
		weight[j] = weight[j]/sum * 1000;
	}*/
}

float dist(float* feat1, float* feat2, float* weight) {
	float toret = 0;
	for (int i = 0; i < FEAT_NUM; i++) {
		if (feat1[i] != -1 and feat2[i] != -1) {
			toret += weight[i] * abs(feat1[i] - feat2[i]);
		}
	}
	return toret;
}

void alg_recommend2(float** feat, float* weight, int start, int end) {

	float* distlist = new float[SITE_NUM * INST_NUM];
	int* recogoodlist = new int[RECOPOINTS_NUM];
	int* recobadlist = new int[RECOPOINTS_NUM];

	for (int i = start; i < end; i++) {
		printf("\rLearning distance... %d (%d-%d)", i, start, end);
		fflush(stdout);
		int cur_site = i/INST_NUM;
		int cur_inst = i % INST_NUM;

		float pointbadness = 0;
		float maxgooddist = 0;

		for (int k = 0; k < SITE_NUM*INST_NUM; k++) {
			distlist[k] = dist(feat[i], feat[k], weight);
		}
		float max = *max_element(distlist, distlist+SITE_NUM*INST_NUM);
		distlist[i] = max;
		for (int k = 0; k < RECOPOINTS_NUM; k++) {
			int ind = min_element(distlist+cur_site*INST_NUM, distlist+(cur_site+1)*INST_NUM) - distlist;
			if (distlist[ind] > maxgooddist) maxgooddist = distlist[ind];
			distlist[ind] = max;
			recogoodlist[k] = ind;
		}
		for (int k = 0; k < INST_NUM; k++) {
			distlist[cur_site*INST_NUM+k] = max;
		}
		for (int k = 0; k < RECOPOINTS_NUM; k++) {
			int ind = min_element(distlist, distlist+ SITE_NUM * INST_NUM) - distlist;
			if (distlist[ind] <= maxgooddist) pointbadness += 1;
			distlist[ind] = max;
			recobadlist[k] = ind;
		}

		pointbadness /= float(RECOPOINTS_NUM);
		pointbadness += 0.2;
		/*
		if (i == 0) {
			float gooddist = 0;
			float baddist = 0;
			printf("Current point: %d\n", i);
			printf("Bad points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recobadlist[k], dist(feat[i], feat[recobadlist[k]], weight));	
				baddist += dist(feat[i], feat[recobadlist[k]], weight);
			}

			printf("Good points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recogoodlist[k], dist(feat[i], feat[recogoodlist[k]], weight));
				gooddist += dist(feat[i], feat[recogoodlist[k]], weight);
			}

			printf("Total bad distance: %f\n", baddist);
			printf("Total good distance: %f\n", gooddist);
		}*/

		float* featdist = new float[FEAT_NUM];
		for (int f = 0; f < FEAT_NUM; f++) {
			featdist[f] = 0;
		}
		int* badlist = new int[FEAT_NUM];
		int minbadlist = 0;
		int countbadlist = 0;
		//printf("%d ", badlist[3]);
		for (int f = 0; f < FEAT_NUM; f++) {
			if (weight[f] == 0) badlist[f] == 0;
			else {
			float maxgood = 0;
			int countbad = 0;
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				float n = abs(feat[i][f] - feat[recogoodlist[k]][f]);
				if (feat[i][f] == -1 or feat[recobadlist[k]][f] == -1) 
					n = 0;
				if (n >= maxgood) maxgood = n;
			}
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				float n = abs(feat[i][f] - feat[recobadlist[k]][f]);
				if (feat[i][f] == -1 or feat[recobadlist[k]][f] == -1) 
					n = 0;
				//if (f == 3) {
				//	printf("%d %d %f %f\n", i, k, n, maxgood);
				//}
				featdist[f] += n;
				if (n <= maxgood) countbad += 1;
			}
			badlist[f] = countbad;
			if (countbad < minbadlist) minbadlist = countbad;	
			}
		}

		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] != minbadlist) countbadlist += 1;
		}
		int* w0id = new int[countbadlist];
		float* change = new float[countbadlist];

		int temp = 0;
		float C1 = 0;
		float C2 = 0;
		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] != minbadlist) {
				w0id[temp] = f;
				change[temp] = weight[f] * 0.01 * badlist[f]/float(RECOPOINTS_NUM) * pointbadness;
				//if (change[temp] < 1.0/1000) change[temp] = weight[f];
				C1 += change[temp] * featdist[f];
				C2 += change[temp];
				weight[f] -= change[temp];
				temp += 1;
			}
		}

		/*if (i == 0) {
			printf("%d %f %f\n", countbadlist, C1, C2);
			for (int f = 0; f < 30; f++) {
				printf("%f %f\n", weight[f], featdist[f]);
			}
		}*/
		float totalfd = 0;
		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] == minbadlist and weight[f] > 0) {
				totalfd += featdist[f];
			}
		}

		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] == minbadlist and weight[f] > 0) {
				weight[f] += C1/(totalfd);
			}
		}

		/*if (i == 0) {
			printf("%d %f %f\n", countbadlist, C1, C2);
			for (int f = 0; f < 30; f++) {
				printf("%f %f\n", weight[f], featdist[f]);
			}
		}*/

		/*if (i == 0) {
			float gooddist = 0;
			float baddist = 0;
			printf("Current point: %d\n", i);
			printf("Bad points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recobadlist[k], dist(feat[i], feat[recobadlist[k]], weight));	
				baddist += dist(feat[i], feat[recobadlist[k]], weight);
			}

			printf("Good points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recogoodlist[k], dist(feat[i], feat[recogoodlist[k]], weight));
				gooddist += dist(feat[i], feat[recogoodlist[k]], weight,);
			}

			printf("Total bad distance: %f\n", baddist);
			printf("Total good distance: %f\n", gooddist);
		}*/
		delete[] featdist;
		delete[] w0id;
		delete[] change;
		delete[] badlist;
	}


	for (int j = 0; j < FEAT_NUM; j++) {
		if (weight[j] > 0)
			weight[j] *= (0.9 + (rand() % 100) / 500.0);
	}
	printf("\n");
	delete[] distlist;
	delete[] recobadlist;
	delete[] recogoodlist;



}

void accuracy(float** trainclosedfeat, float** testclosedfeat, float** openfeat, float* weight, float & tp, float & tn) {

	//trainfeat is the "background" knn closed points.
	//testfeat is the possibly modified knn closed points that we want to test accuracy on
	//testfeat = trainfeat is normal
	//openfeat is the knn open points. they are also tested. 

	tp = 0;
	tn = 0;

	float** trainfeat = new float*[SITE_NUM*TEST_NUM + OPENTEST_NUM];
	float** testfeat = new float*[SITE_NUM*TEST_NUM + OPENTEST_NUM];

	for (int i = 0; i < SITE_NUM*TEST_NUM; i++) {
		trainfeat[i] = trainclosedfeat[i];
	}
	for (int i = 0; i < OPENTEST_NUM; i++) {
		trainfeat[i + SITE_NUM * TEST_NUM] = openfeat[i];
	}

	for (int i = 0; i < SITE_NUM*TEST_NUM; i++) {
		testfeat[i] = testclosedfeat[i];
	}
	for (int i = 0; i < OPENTEST_NUM; i++) {
		testfeat[i + SITE_NUM * TEST_NUM] = openfeat[i];
	}

	float* distlist = new float[SITE_NUM * TEST_NUM + OPENTEST_NUM];
	int* classlist = new int[SITE_NUM + 1];

	float* opendistlist = new float[OPENTEST_NUM];

	FILE * logfile;
	logfile = fopen("flearner.log", "w");

	for (int is = 0; is < SITE_NUM*TEST_NUM + OPENTEST_NUM; is++) {
		printf("\rComputing accuracy... %d (%d-%d)", is, 0, SITE_NUM*TEST_NUM + OPENTEST_NUM);
		fflush(stdout);
		for (int i = 0; i < SITE_NUM+1; i++) {
			classlist[i] = 0;
		}
		int maxclass = 0;
		for (int at = 0; at < SITE_NUM * TEST_NUM + OPENTEST_NUM; at++) {
			distlist[at] = dist(testfeat[is], trainfeat[at], weight);
		}
		float max = *max_element(distlist, distlist+SITE_NUM*TEST_NUM+OPENTEST_NUM);
		distlist[is] = max;
			
		fprintf(logfile, "Guessed classes: ");
		for (int i = 0; i < NEIGHBOUR_NUM; i++) {
			int ind = find(distlist, distlist + SITE_NUM*TEST_NUM+OPENTEST_NUM, *min_element(distlist, distlist+SITE_NUM*TEST_NUM+OPENTEST_NUM)) - distlist;
			int classind = 0;
			if (ind < SITE_NUM * TEST_NUM) {
				classind = ind/TEST_NUM;
			}
			else {
				classind = SITE_NUM;
			}
			classlist[classind] += 1;
			
			fprintf(logfile, "%d ", classind);

			if (classlist[classind] > maxclass) {
				maxclass = classlist[classind];
			}
			distlist[ind] = max;
		}

		int trueclass = is/TEST_NUM;
		if (trueclass > SITE_NUM) trueclass = SITE_NUM;

		fprintf(logfile, ", True class: %d\n", trueclass);

		int countclass = 0;
		int hascorrect = 0;

		int hasconsensus = 0;
		for (int i = 0; i < SITE_NUM+1; i++) {
			if (classlist[i] == NEIGHBOUR_NUM) {
				hasconsensus = 1;
			}
		}
		if (hasconsensus == 0) {
			for (int i = 0; i < SITE_NUM; i++) {
				classlist[i] = 0;
			}
			classlist[SITE_NUM] = 1;
			maxclass = 1;
		}

		for (int i = 0; i < SITE_NUM+1; i++) {
			if (classlist[i] == maxclass) {
				countclass += 1;
				if (i == trueclass) {
					hascorrect = 1;
				}
			}
		}

		float thisacc = 0;
		if (hascorrect == 1) {
			thisacc = 1.0/countclass;
		}
		if (trueclass == SITE_NUM) {
			tn += thisacc;
		}
		else { 
			tp += thisacc;
		}
		
	}

	fclose(logfile);

	printf("\n");

	delete[] distlist;
	delete[] classlist;
	delete[] opendistlist;
	delete[] trainfeat;
	delete[] testfeat;
	
	tp /= SITE_NUM*TEST_NUM;
	if (OPENTEST_NUM > 0)	tn /= OPENTEST_NUM;
	else tn = 1;
}

void readfile(string folder, float ** feat, int SITE_NUM, int INST_NUM, int is_OPEN) {
	//if is_OPEN, then INST_NUM is always 1

	if (is_OPEN == 1) {
		INST_NUM == 1;
	}

	for (int cur_site = 0; cur_site < SITE_NUM; cur_site++) {
		int fail_count = 0;
		for (int cur_inst = 0; cur_inst < INST_NUM; cur_inst++) {
			int gotfile = 0;
			ifstream fread;

			while (gotfile == 0) {
				ostringstream freadnamestream;
				if (is_OPEN != 1) {
					freadnamestream << folder << cur_site << "-" << cur_inst + fail_count << ".sizef";
				}
				if (is_OPEN == 1) {
					freadnamestream << folder << cur_site << ".sizef";
				}
				string freadname = freadnamestream.str();
				fread.open(freadname.c_str());
				if (fread.is_open() and fread.peek() != EOF) {
					gotfile = 1;
				}
				else {
					fail_count += 1;
				}
			}
			string str = "";
			getline(fread, str);
			fread.close();

			string tempstr = "";
			int feat_count = 0;
			for (int i = 0; i < str.length(); i++) {
				if (str[i] == ' ') {
					if (tempstr.c_str()[1] == 'X') {
						feat[cur_site * INST_NUM + cur_inst][feat_count] = -1;
					}
					else {
						feat[cur_site * INST_NUM + cur_inst][feat_count] = atof(tempstr.c_str());
					}	
					feat_count += 1;
					tempstr = "";
				}
				else {
					tempstr += str[i];
				}
			}
		}
	}
}


int main(int argc, char** argv) {
	/*int OPENTEST_list [6] = {100, 500, 1000, 3000, 5000, 6000};
	int NEIGHBOUR_list [10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	if(argc == 3){
		int OPENTEST_ind = atoi(argv[1]); 
		int NEIGHBOUR_ind = atoi(argv[2]);

		OPENTEST_NUM = OPENTEST_list[OPENTEST_ind % 5];
		NEIGHBOUR_NUM = NEIGHBOUR_list[NEIGHBOUR_ind % 10];
	}

	srand(time(NULL));*/

	float** feat = new float*[SITE_NUM*INST_NUM];
	float** trainclosedfeat = new float*[SITE_NUM*TEST_NUM];
	float** testclosedfeat = new float*[SITE_NUM*TEST_NUM];
	float** openfeat = new float*[OPENTEST_NUM];

	for (int i = 0; i < SITE_NUM*INST_NUM; i++) {
		feat[i] = new float[FEAT_NUM]; //for weight learning
	}
	for (int i = 0; i < SITE_NUM*TEST_NUM; i++) {
		trainclosedfeat[i] = new float[FEAT_NUM]; //for testing/training
	}
	for (int i = 0; i < SITE_NUM*TEST_NUM; i++) {
		testclosedfeat[i] = new float[FEAT_NUM]; //for testing/training
	}
	for (int i = 0; i < OPENTEST_NUM; i++) {
		openfeat[i] = new float[FEAT_NUM]; //open elements for testing/training
	}

	readfile(folder, feat, SITE_NUM, INST_NUM, 0);
	printf("Main Instances loaded\n");
	readfile(foldertrain, trainclosedfeat, SITE_NUM, TEST_NUM, 0);

	printf("Training Instances loaded\n");
	readfile(foldertest, testclosedfeat, SITE_NUM, TEST_NUM, 0);

	printf("Testing Instances loaded\n");
	readfile(folderopen, openfeat, OPENTEST_NUM, 1, 1);

	printf("Open Instances loaded\n");

	float * weight = new float[FEAT_NUM];
	float * value = new float[FEAT_NUM];

	alg_init_weight(feat, weight);

	float * prevweight = new float[FEAT_NUM];
	for (int i = 0; i < FEAT_NUM; i++) {
		prevweight[i] = weight[i];
	}

	alg_init_weight(feat, weight);
	alg_recommend2(feat, weight, 0, SITE_NUM * INST_NUM);
	float tp, tn;
	accuracy(trainclosedfeat, testclosedfeat, openfeat, weight, tp, tn);
	printf("Accuracy: %f %f\n", tp, tn);

	FILE * weightfile;
	weightfile = fopen("weights", "w");
	for (int i = 0; i < FEAT_NUM; i++) {
		fprintf(weightfile, "%f ", weight[i] * 1000);
	}
	fclose(weightfile);

	for (int i = 0; i < SITE_NUM * INST_NUM; i++) {
		delete[] feat[i];
	}
	delete[] feat;
	for (int i = 0; i < SITE_NUM * TEST_NUM; i++) {
		delete[] trainclosedfeat[i];
	}
	delete[] trainclosedfeat;
	for (int i = 0; i < SITE_NUM * TEST_NUM; i++) {
		delete[] testclosedfeat[i];
	}
	delete[] testclosedfeat;
	for (int i = 0; i < OPENTEST_NUM; i++) {
		delete[] openfeat[i];
	}
	delete[] openfeat;

	delete[] prevweight;
	delete[] weight;
	delete[] value;
	return 0;
}
