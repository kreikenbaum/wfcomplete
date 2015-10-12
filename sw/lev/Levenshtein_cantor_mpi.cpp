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
	TRACE* str1;
	TRACE* str2;
	int websites;
	int trials;
	int METRIC;
	int METHOD;

public:
	Levenshtein(const char* folder, int metr, int WEBSITES, int TRIALS);
	~Levenshtein();
	CORD inverse_cantor(int z);
	
	int Parse_data(char* fname);
	TRACE* fetch_pool(int web, int trial);
//	void Pre_process();

	double DLdis(int ms, int ns);
	double minimum(double a, double b, double c);

	int DLdel(int ms, int ns, int delele);		
	double DLtranscost(int* str1, int* str2, int ms, int ns);
	double newDLdis(int ms, int ns);
};


Levenshtein :: Levenshtein(const char* folder, int metr, int WEBSITES, int TRIALS){
	pool.clear();
	buffer.clear();
	sizes.clear();
	str1 = str2 = NULL;

	METRIC = metr;

	METHOD = 8;
	char fname[200];

	websites = WEBSITES;
	trials = TRIALS;

	for(int web = 1; web <= websites; web++){
		for(int trial = 1; trial <= trials; trial++){
			memset(fname, 0, 200);
			sprintf(fname,"%s%d_%d.txt", folder,web,trial);
			Parse_data(fname);
		}
	}

	cout<<"constructor finished, size of pool is: "<<pool.size()<<endl;
}

Levenshtein :: ~Levenshtein(){
	for(int i = 0; i < pool.size();i++)
		delete[] pool.at(i);
}

TRACE* Levenshtein::fetch_pool(int web, int trial){
	int index = (web-1)*trials+trial-1;
	if(index >= pool.size()){
		cout<<"fetching something out of pool, error! -- web "<<web<<", trial "<<trial<<endl;
		exit(1);
	}
	return pool.at(index);
}


int Levenshtein::Parse_data(char *fname){
	FILE* fp = NULL;
	int length,size;
	int i, round;
	
	buffer.clear();
	buffer.push_back(0);

	fp = fopen(fname, "r");
	if(!fp){
		cout<<fname<<"  cannot open!"<< errno <<endl;
		return -1;
	}
	while(!feof(fp)){
		if(0 > fscanf(fp,"%d",&length))
			continue;
		if(abs(length) > 1500)
			continue;
		if(abs(length) <= 84)
			continue;
		length = length/abs(length) * INCREMENT * (int)ceil(abs(length)/1.0/INCREMENT);
		buffer.push_back(length);
	}	
	
	fclose(fp);

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
//METRICCLASS 1: ORIGINAL (combine with round to 600)
//METRICCLASS 2: DISABLE SUBSTITUTION
//METRICCLASS 3: SUBSTITUTION/INSERTING POSITIVE COSTS 3 TIMES AS MUCH
//METRICCLASS 4: TRANSPOSITION
//METRICCLASS 5: SOME COMBINATION
//METRICCLASS 6: DL
	double ret = 0;
	int min;
//	Pre_process();

//	printf("BEGIN\n");

	int m = ms;
	int n = ns;
	min = m < n ? m : n;
	min = min == 0 ? 1 : min;

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
	int* dicn;
	dicn = new int[maxpacket-minpacket +1];
	for (i = 0; i < maxpacket-minpacket +1; i++) {
		dicn[i] = -1;
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
	
	int db, x1, y1 = 0;
	float P = 0;

	int METRICCLASS = METRIC/10;

	float idcost[2] = {2, 2};
	
	if (METRICCLASS == 3 || METRICCLASS == 5)
		idcost[0] = 6;
	
	if (METRICCLASS == 2 || METRICCLASS == 5)
		subcost = 20;
	else
		subcost = 2;

	if (METRICCLASS == 6)
		transcost = 2;
	else
		transcost = 0.1;

	for(i=1; i<m; i++){
		db = 0;
		for(j=1; j<n; j++){
			if (METRICCLASS == 4 || METRICCLASS == 5) {
				P = (float)i/m > (float)j/n ? (float)j/n : (float)i/m;
				transcost = (1-P*(0.9)); //goes from 1 to 0.1;
				transcost *= transcost;
				//printf("%f %f\n", P, transcost);
			}
			y1 = db;
			x1 = dicn[str2[j]-minpacket];
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
				dis[i][j] = minimum(
				dis[i-1][j] + idcost[(str1[i] > 0 ? 0 : 1)], //abs(str1[i]),  // a deletion
        		        dis[i][j-1] + idcost[(str2[j] > 0 ? 0 : 1)], //abs(str2[j]),  // an insertion
        		        dis[i-1][j-1] + subcost); // a substitution
			}
	
			if (METRICCLASS == 6) {
				if (x1 >= 0 && y1 >= 0) {
					//printf("%f %f\n", dis[i][j], dis[x1-1][y1-1] + transcost + ((i - x1-1) + (j - y1-1))*idcost);
					dis[i][j] = 	dis[i][j] < dis[x1-1][y1-1] + transcost + ((i - x1-1) + (j - y1-1))*idcost[1] ? 
							dis[i][j] : dis[x1-1][y1-1] + transcost + ((i - x1-1) + (j - y1-1))*idcost[1];	
				}
			}
			else {
				if(i > 1 && j > 1 && str1[i] == str2[j-1] && str1[i-1] == str2[j]) {
					dis[i][j] = 	dis[i][j] < dis[i-2][j-2] + transcost ? 
							dis[i][j] : dis[i-2][j-2] + transcost;
				}
			}
			//printf("%d %d %d %d \n", i, j, dis[i][j], dis);
		}
		dicn[str1[i]-minpacket] = i;
	}
//	printf("END: %f %d\n", dis[m-1][n-1], min);
	ret = dis[m-1][n-1]/min;

	for(i = 0 ; i < m; i++) {
		delete[] dis[i];
	}
	delete[] dis;
	delete[] dicn;

	return ret;
}

int abs(int k) {
	if (k > 0)
		return k;
	else
		return -k;
}

double Levenshtein::newDLdis(int ms, int ns) {
	double transcost = (1+(METRIC % 5))*0.01;
	double posdelcost = (METRIC/5)/2.0;
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

	if(argc != 8){
	    cout <<"example: ./Levenshtein_cantor_mpi <traces folder> <#websites> <#trials> <metric number> <job portion> <total number> <output name>"<<endl;
	    exit(1);
	}

	FILE *fp = NULL;
	double dist = -1;
	CORD ret;
	int i,j,begin,end;

	double aug = 1;	
	const char* folder = argv[1];
	int WEBSITES = atoi(argv[2]);
	int TRIALS = atoi(argv[3]);
	int metr = atoi(argv[4]); 
	int round = atoi(argv[5]);
	int CORES = atoi(argv[6]);
	const char* outname = argv[7]; 

	Levenshtein Lclass(folder, metr, WEBSITES, TRIALS);
		
// parameters need to be modified 
	const char* prefix = "./cantor_Tor_100_40_";
	int dim = Lclass.websites*Lclass.trials;
	int total = (dim*dim-dim)/2/CORES; // each node's task

	fp = fopen(outname, "a+");
	if(!fp){
		cout<<"cannot open file "<<outname<<" !"<<endl;
		exit(1);
	}

	begin = round*total;	// round == 0,1,2, ... ,5
	end = begin+total-1;

	if (round == CORES - 1) { //the last one
		end = (dim*dim-dim)/2 -1; //to finish all the remaining jobs; only 0.1% more jobs for 256 cores 100 sites 40 instances per. 
	}

	int web_x,trial_x,web_y,trial_y;
	int ms,ns;
	clock_t t1, t2;
	t1 = clock();

	printf("Total job: %d - %d\n", begin, end);

	for(int index = begin; index <= end; index++){
		//moi
		//if (index %3000 == 0)
		//	printf("Handling job: %d\n", index);
		
		ret = Lclass.inverse_cantor(index);
		web_x = (ret.x/Lclass.trials) + 1;
		trial_x = (ret.x%Lclass.trials) + 1;

		web_y = ((dim-1-ret.y)/Lclass.trials) + 1;
		trial_y = ((dim-1-ret.y)%Lclass.trials) + 1;
		
		Lclass.str1 = Lclass.fetch_pool(web_x,trial_x);
		Lclass.str2 = Lclass.fetch_pool(web_y,trial_y);
		
		ms = Lclass.sizes.at((web_x-1)*Lclass.trials+trial_x-1);
		ns = Lclass.sizes.at((web_y-1)*Lclass.trials+trial_y-1);
		
		dist = Lclass.DLdis(ms, ns); //change this line to dist = LClass.newDLdis(ms, ns) to run our fast Levenshtein-like alg. Note how METRIC works in the code. 
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


