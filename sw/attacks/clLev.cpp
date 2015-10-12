//#include <mpi.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <time.h>

#include <iostream>
using namespace std;

#define UNIT 512
#define INCREMENT 600

typedef struct _cord{
	int x;
	int y;
}CORD;

typedef int TRACE;

class Levenshtein{

public:
	vector<int> buffer;
	vector<int> sizes;
	vector<TRACE*> pool;
	vector <int> torsize;
	TRACE* str1;
	TRACE* str2;
	int websites;
	int trials;
	int ndistr[1501];
	int pdistr[1501];
	int METRIC;
	int METHOD;

public:
	Levenshtein(const char* folder, int metr);
	~Levenshtein();
	CORD inverse_cantor(int z);

	void get_distr(char* fname);
	int get_sprime(int length);
	int augment(vector<int>& trace, double aug);
	
	int Parse_data(char* fname, int type, double aug);
	TRACE* fetch_pool(int web, int trial);
	double DLdis(int ms, int ns);
	double minimum(double a, double b, double c);

	int DLdel(int ms, int ns, int delele);		
	double DLtranscost(int* str1, int* str2, int ms, int ns);
	double newDLdis(int ms, int ns);
};


Levenshtein :: Levenshtein(const char* folder, int metr){
	
	double aug = 1;
	torsize.clear();
	pool.clear();
	buffer.clear();
	sizes.clear();
	str1 = str2 = NULL;

	METRIC = metr;

	METHOD = 8;
	char fname[200];

	websites = 100;
	trials = 40;

//	if(METHOD == SAMPLE_MORPHING)
//		get_distr("./log_ssh/23_1.txt");
	
	//const char* folder = "./celltraces/";	//this folder holds all the cell trace files //moi
	for(int web = 1; web <= websites; web++){

		for(int trial = 1; trial <= trials; trial++){
			memset(fname, 0, 200);
			sprintf(fname,"%s%d-%d.size", folder,web-1,trial-1);
			Parse_data(fname, METHOD, aug);
		}
	}

	cout<<"constructor finished, size of pool is: "<<pool.size()<<endl;
}

Levenshtein :: ~Levenshtein(){
	for(int i = 0; i < pool.size();i++)
		delete[] pool.at(i);
}

void Levenshtein::get_distr(char* fname){
	int length;
	FILE* fp;
	if((fp = fopen(fname, "r")) == NULL){
		printf("%s cannot be opened!\n", fname);
		exit(1);
	}
	for(int i = 0; i < 1501; i++)
		pdistr[i] = ndistr[i] = 0;
 
	while(!feof(fp)){
		if(0 > fscanf(fp, "%d", &length))
			continue;
		if(abs(length) > 1500)
			continue;

#if REMOVE_ACK
		if(abs(length) <= 84)
			continue;
#endif
		if(length < 0)
			ndistr[abs(length)]++;
		else
			pdistr[length]++;
	}
	fclose(fp);
	
	for(int i = 1; i < 1501; i++){
		pdistr[i] += pdistr[i-1];
		ndistr[i] += ndistr[i-1];
	}
}


int Levenshtein::get_sprime(int length){
	int max, picked, i;
	if(length >= 0){
		max = pdistr[1500];
		picked = rand()%max+1;
		for(i = 0; i < 1501 && pdistr[i] < picked; i++);
		picked = i;
		return picked;
	}
	else{
		max = ndistr[1500];
		picked = rand()%max+1;
		for(i = 1; i < 1501 && ndistr[i] < picked; i++);
		picked = -i;
		return picked;
	}
}

int Levenshtein::augment(vector<int>& trace, double aug){
	int tsize = trace.size();
	int size = (int)ceil(tsize*aug);
	int extrasize = size-tsize;
	int x,y,i,j;
	vector<int> extra;
	vector<int> copy;
	
	srand(time(NULL));
	if(extrasize == 0)
		return tsize;

	for(i = 0; i < tsize; i++)
		copy.push_back(trace.at(i));

	for(i = 0 ; i < extrasize; i++){
		x = rand()%2;
		extra.push_back(x == 0 ? -1 : 1);
	}

	trace.clear();
	// begin to merge copy and extra to trace

	for(i = j = 0; i < tsize && j < extrasize;){
			y = rand()%size;
			if(y < tsize)
				trace.push_back(copy.at(i++));
			else
				trace.push_back(extra.at(j++));
	}
	while(i < tsize)
		trace.push_back(copy.at(i++));
	while(j < extrasize)
		trace.push_back(extra.at(j++));

	return size;
}


TRACE* Levenshtein::fetch_pool(int web, int trial){
	int index = (web-1)*trials+trial-1;
	if(index >= pool.size()){
		cout<<"fetching something out of pool, error! -- web "<<web<<", trial "<<trial<<endl;
		exit(1);
	}
	return pool.at(index);
}


int Levenshtein::Parse_data(char *fname, int type, double aug){
	FILE* fp = NULL;
	int length,size;
	double time;
	int i, round;
	
	buffer.clear();
	buffer.push_back(0);

	fp = fopen(fname, "r");
	if(!fp){
		cout<<fname<<"  cannot open!"<< errno <<endl;
		return -1;
	}
	
	while(!feof(fp)){
		if(0 > fscanf(fp,"%f\t%d",&time, &length))
			continue;
		length = length/abs(length); //* INCREMENT * (int)ceil(double(abs(length))/INCREMENT);
		buffer.push_back(length);
	}
	
	fclose(fp);

//	size = augment(buffer, aug);
	TRACE* tmp = new TRACE[buffer.size()];
	for(int x = 0; x < buffer.size(); x++)
		tmp[x] = buffer.at(x);
	pool.push_back(tmp);
	sizes.push_back(buffer.size());
	return 0;
}

int contains(int* str1, int element, int len) {
	int toret = -1;
	for (int i = 0; i < len; i++) {
		if (str1[i] == element)
			toret = i;
	}
	return toret;
}

double Levenshtein::DLdis(int ms, int ns){
//METRIC 1: ORIGINAL (combine with round to 600)
//METRIC 2: DISABLE SUBSTITUTION
//METRIC 3: SUBSTITUTION/INSERTING POSITIVE COSTS 3 TIMES AS MUCH
//METRIC 4: TRANSPOSITION
//METRIC 5: SOME COMBINATION
//METRIC 6: DL
//METRIC 7: DL + SOME COMBINATION
	double ret = 0;
	int min;
//	Pre_process();

	int m = ms;
	int n = ns;
	min = m < n ? m : n;
	min = min == 0 ? 1 : min;

	if (METRIC == 7) {
		int str1pos, str1neg, str2pos, str2neg = 0;
		for (int i = 0; i < ms; i++) {
			if (str1[i] > 0)
				str1pos += 1;
			if (str1[i] < 0)
				str1neg += 1;
		}
		for (int i = 0; i < ns; i++) {
			if (str2[i] > 0)
				str2pos += 1;
			if (str2[i] < 0)
			str2neg += 1;
			}

		int dif1 = str1pos - str2pos;
		int dif2 = str1neg - str2neg;
	
		if (dif1 < 0)
			dif1 = -dif1;
		if (dif2 < 0)
			dif2 = -dif2;
	
		double weight = 0.5;
	
		return (float)(dif1 + weight * dif2)/min;
	}

	int i,j;
    	float subcost,transcost;

	float** dis = new float*[m];
	for(i=0;i<m;i++)
		dis[i]= new float[n];

	int maxpacket = 0;
	int minpacket = 0;

	for (int k = 0; k < 2; k++) {
		int* pt;
		int len;
		if (k == 0) {
			pt = str1;
			len = m;
		}
		if (k == 1) {
			pt = str2;
			len = n;
		}
		for (i = 0; i < len; i++) {
			if (pt[i] > maxpacket)
				maxpacket = pt[i];
			if (pt[i] < minpacket)
				minpacket = pt[i];
		}
	}
/*
	for(i = 0; i < m ;i++)
		for(j = 0; j < n; j++) 
			dis[i][j] = -1;
*/
	for(i=0; i<m; i++)
		dis[i][0]=i*2;
	for(j=0; j<n; j++)
		dis[0][j]=j*2;
	
	int db, x1, y1, P = 0;

	float idcost[2] = {2, 2};
	
	if (METRIC == 2)
		idcost[0] = 6;
	
	if (METRIC == 2)
		subcost = 20;
	else
		subcost = 2;

	if (METRIC == 1)
		transcost = 0.1;

	for(i=1; i<m; i++){
		db = 0;
		for(j=1; j<n; j++){
			if (METRIC == 2) {
				P = (float)i/m > (float)j/n ? (float)j/n : (float)i/m;
				transcost = (1-P*0.9) * (1-P*0.9); //goes from 1 to 0.01;
			}
			//printf("%d %d %d %d %d\n", METRIC, i, j, x1, y1);
			if (str1[i] == str2[j]) {
				dis[i][j] = minimum (
					dis[i-1][j] + idcost[(str1[i] > 0 ? 0 : 1)], //abs(str1[i]),  // a deletion
        	        		dis[i][j-1] + idcost[(str2[j] > 0 ? 0 : 1)], //abs(str2[j]),  // an insertion
        	        		dis[i-1][j-1] // a substitution
				);
				db = j;
			}
			else {
				dis[i][j] = minimum (
					dis[i-1][j] + idcost[(str1[i] > 0 ? 0 : 1)], //abs(str1[i]),  // a deletion
        	 	       		dis[i][j-1] + idcost[(str2[j] > 0 ? 0 : 1)], //abs(str2[j]),  // an insertion
        		        	dis[i-1][j-1] + subcost // a substitution
				);
				if(i > 1 && j > 1 && str1[i] == str2[j-1] && str1[i-1] == str2[j]) {
					dis[i][j] = 	dis[i][j] < dis[i-2][j-2] + transcost ? 
							dis[i][j] : dis[i-2][j-2] + transcost;
				}
			}
			//printf("%d %d %d %d \n", i, j, dis[i][j], dis);
		}
	}
	ret = dis[m-1][n-1]/min;

	for(i = 0 ; i < m; i++) {
		delete[] dis[i];
	}
	delete[] dis;

	return ret;
}

int abs(int k) {
	if (k > 0)
		return k;
	else
		return -k;
}

double Levenshtein::newDLdis(int ms, int ns) {
	//from transcost = 0.01, 0.02, 0.03 x posdelcost = 2, 4, 6, 4 * 0.01 is the best
//	double transcost = (1+(METRIC))*0.001;
	double transcost = 0.01;
	double posdelcost = 4;
	double negdelcost = 1;
	int i = 0;
	int poscount = 0;
	int negcount = 0;
	int min = 0;

	min = ms < ns ? ms : ns;
	min = min == 0 ? 1 : min;

	for (i = 0; i < ms; i++) {
		if (str1[i] > 0)
			poscount += 1;
		else
			negcount += 1;
	}
	
	for (i = 0; i < ns; i++) {
		if (str2[i] > 0)
			poscount += 1;
		else
			negcount += 1;
	}

	if (ns == 0) {
		return 0;
	}
	int* dicn = new int[ms]; //dictionary of all elements in str1
	int dicnlen = 0; //true size of dictionary; saves memory. 
	for (i = 0; i < ms; i++) {
		int a = contains(dicn, str1[i], dicnlen);
		if (a == -1) {
			dicn[dicnlen] = str1[i];
			dicnlen += 1;
		}
	}

	int** dicncount = new int*[dicnlen]; //location of all elements in str1
	int* dicncountlen = new int[dicnlen];
	int* dicncountlentemp = new int[dicnlen];
	int count = 0;
	int dist = 0;

	for (i = 0; i < dicnlen; i++) {
		dicncount[i] = new int[ms];
		dicncountlen[i] = 0;
	}
	for (i = 0; i < ms; i++) {
		int a = contains(dicn, str1[i], dicnlen);
		dicncount[a][dicncountlen[a]] = i;
		dicncountlen[a] += 1;
	}
	for (i = 0; i < dicnlen; i++) {
		dicncountlentemp[i] = dicncountlen[i];
	}


	for (i = 0; i < ns; i++) {
		int a = contains(dicn, str2[i], dicnlen);
		if (a != -1) {
			count = dicncountlen[a] - dicncountlentemp[a];
			if (count < dicncountlen[a]) {
				if (str2[i] > 0)
					dist += (dicncount[a][count] - i > 0)? dicncount[a][count] - i : i - dicncount[a][count];
				dicncountlentemp[a] -= 1;
				if (str2[i] > 0)
					poscount -= 1;
				else
					negcount -= 1;
			}
		}
	}

	double cost = poscount * posdelcost + negcount * negdelcost + dist * transcost;

	for (int i = 0; i < dicnlen; i++) {
		delete[] dicncount[i];
	}

	delete[] dicncount;
	delete[] dicn;
	delete[] dicncountlen;
	delete[] dicncountlentemp;

	return cost/min;

}

double Levenshtein::minimum(double a, double b, double c){
	double min = a;
	if(b < min)
		min = b;
	if(c < min)
		min = c;

	return min;
}

CORD Levenshtein::inverse_cantor(int z){
	CORD ret;
	int w = floor((sqrt(8.0*z+1)-1)/2);
	int t = (w*w+w)/2;
	ret.y = z-t;
	ret.x = w-ret.y;

	return ret;
}

int main(int argc, char** argv){
//

	if(argc != 6){
	    cout <<"example: ./clLev <traces folder> <method number> <output name> <corenum> <coretotal>"<<endl;
		cout << "method 1 = Ca-DLevenshtein, method 2 = Wa-OSAD, method 3 = Wa-fastLevenshtein" << endl;
	    exit(1);
	}

	FILE *fp = NULL;
	double dist = -1;
	CORD ret;
	int i,j,begin,end;

	double aug = 1;	
	const char* folder = argv[1];
	int metr = atoi(argv[2]);
	const char* outname = argv[3]; 
	int round = atoi(argv[4]);
	int CORES = atoi(argv[5]);

	Levenshtein Lclass(folder, metr);

	int dim = Lclass.websites*Lclass.trials;
	int total = (dim*dim-dim)/2/CORES; // each node's task

	printf("Total: %d\n", total);
	printf("dim: %d\n", dim);

	fp = fopen(outname, "a+");
	if(!fp){
		cout<<"cannot open file "<<outname<<" !"<<endl;
		exit(1);
	}
/*
	begin = 10*(round-1)+round;
	end = (round == 10) ? websites : round*11;
*/

	begin = round*total;
	end = begin+total-1;

	if (round == CORES - 1) { //the last one
		end = (dim*dim-dim)/2 -1; //to finish all the remaining jobs; only 0.1% more jobs for 256 cores 100 sites 40 instances per. 
	}

	int web_x,trial_x,web_y,trial_y;
	int ms,ns;
	clock_t t1, t2, t1p, t2p;
	t1 = clock();

	printf("Total job: %d - %d\n", begin, end);

	for(int index = begin; index <= end; index++){
		//moi
		if (index % 100 == 0) {
			t2 = clock();
			printf("%d\t", index);
			printf("Total time elapsed: %f\n", (float)(t2-t1)/(CLOCKS_PER_SEC));
		}

		
		ret = Lclass.inverse_cantor(index);
		web_x = (ret.x/Lclass.trials) + 1;
		trial_x = (ret.x%Lclass.trials) + 1;

		web_y = ((dim-1-ret.y)/Lclass.trials) + 1;
		trial_y = ((dim-1-ret.y)%Lclass.trials) + 1;
		
		Lclass.str1 = Lclass.fetch_pool(web_x,trial_x);
		Lclass.str2 = Lclass.fetch_pool(web_y,trial_y);
		
		ms = Lclass.sizes.at((web_x-1)*Lclass.trials+trial_x-1);
		ns = Lclass.sizes.at((web_y-1)*Lclass.trials+trial_y-1);
		
		if (metr == 1 || metr == 2) {
			dist = Lclass.DLdis(ms, ns);
		}
		if (metr == 3) {
			dist = Lclass.newDLdis(ms, ns);
		}
		if (dist != -1){
			fprintf(fp,"%d;%d;%d;%d;%d;%d;%lf\n", web_x, trial_x, web_y, trial_y, ms, ns, dist);
			fprintf(fp,"%d;%d;%d;%d;%d;%d;%lf\n", web_y, trial_y, web_x, trial_x, ns, ms, dist);
		}
	}
	
	fclose(fp);
	t2 = clock();

	printf("Average/Total time taken: %f/%f\n", (float)(t2-t1)/(CLOCKS_PER_SEC * (end-begin)), (float)(t2-t1));
	printf("Parameters: %s Metric:%d Jobs:%d/%d Range: %d %d\n", folder, metr, round, CORES, begin, end);

//	cout<<"time to compute "<<end-begin+1<<" distances is: "<<tend-tstart<<" us"<<endl;
	return 0;
}


